# -*- coding: utf-8 -*-
"""
Cálculo dos IPPs (ionospheric pierce points) a partir do dTEC, para TODOS os
pares (year, doy) encontrados nas pastas, no mesmo estilo de
``process_tec_pcarrier.py`` e ``transform_to_dtec.py``.

Para cada dia e cada estação:
    1. lê o parquet de dTEC (prn, dtec_*, x_km, y_km, z_km = posição do satélite);
    2. calcula latitude, longitude e elevação do IPP com ``ipp_calc2``
       (colunas ``lts``, ``lns``, ``els``);
    3. grava ``{estação}_dtecIPP_{year}{doy}.parquet`` em ``dtec_ipp_dir`` com
       ``horario`` como índice (é o formato que o script de keogramas lê).

Estações cujo IPP já existe são puladas, então rodar de novo retoma de onde
parou. A escrita é atômica (.tmp -> rename).

Uso::

    python visualize_ipp.py                       # todos os dias encontrados
    python visualize_ipp.py --year 2021           # um ano
    python visualize_ipp.py --year 2021 --doy 296 # um único dia
    python visualize_ipp.py --overwrite           # refaz mesmo os já gravados

Só gera arquivos; nenhuma figura é criada.

@author: hisas
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from shared.config import station_positions_dir, dtec_ipp_dir, setup_logging
from shared.geometry import ipp_calc2
from shared.pipeline_utils import build_arg_parser, resolve_pairs
from shared.prn_store import get_station_paths

log = logging.getLogger("process.ipp")

# Chave da pasta de referência usada por resolve_pairs() para descobrir os
# pares (year, doy). Mesma dos outros scripts do pipeline.
PAIRS_REFERENCE = "rinexPKLReferenceFolder"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ipp_path(folder: Path, station: str, year: int, doy: int, padded: bool = False) -> Path:
    """Caminho do IPP de uma estação.

    O nome histórico usa o doy SEM zeros ({year}{doy}); ``padded=True`` dá a
    variante com zeros. Para doy >= 100 as duas são idênticas.
    """
    tag = f"{year}{doy:03d}" if padded else f"{year}{doy}"
    return folder / f"{station}_dtecIPP_{tag}.parquet"


def _already_done(folder: Path, station: str, year: int, doy: int) -> bool:
    return (_ipp_path(folder, station, year, doy).exists()
            or _ipp_path(folder, station, year, doy, padded=True).exists())


# ---------------------------------------------------------------------------
# Um dia
# ---------------------------------------------------------------------------

def run(year: int, doy: int, overwrite: bool = False) -> None:
    """Calcula os IPPs de todas as estações de um dia (year, doy)."""
    sp_file = station_positions_dir() / f"SP_{year}-{doy:03d}.parquet"
    if not sp_file.exists():
        log.warning("%d/%03d: arquivo de posições das estações não existe (%s) — dia pulado",
                    year, doy, sp_file)
        return

    dtec_fmap = get_station_paths(year, doy)   # {estação: parquet de dTEC}
    if not dtec_fmap:
        log.warning("%d/%03d: nenhum parquet de dTEC encontrado — dia pulado", year, doy)
        return

    df_stations = pd.read_parquet(sp_file).set_index("name")
    stations = sorted(set(df_stations.index) & set(dtec_fmap.keys()))
    if not stations:
        log.warning(
            "%d/%03d: nenhuma estação em comum (%d com posição, %d com dTEC) — dia pulado",
            year, doy, len(df_stations), len(dtec_fmap),
        )
        return

    out_dir = Path(dtec_ipp_dir(year, doy))
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(stations)
    n_done = n_skipped = n_empty = n_failed = 0

    n_existing = sum(_already_done(out_dir, s, year, doy) for s in stations)
    log.info("%d/%03d: %d estações; %d/%d (%.1f%%) já gravadas",
             year, doy, total, n_existing, total, 100.0 * n_existing / total)

    for i, station in enumerate(stations, start=1):
        pct = 100.0 * i / total  # progresso do dia (inclui as já gravadas)

        if not overwrite and _already_done(out_dir, station, year, doy):
            n_skipped += 1
            log.debug("[%d/%d %5.1f%%] SKIP %s (já gravada)", i, total, pct, station)
            continue

        try:
            df_dtec = pd.read_parquet(dtec_fmap[station]).set_index("horario")

            # Chamada única e vetorizada para todos os PRNs
            x, y, z = df_dtec[["x_km", "y_km", "z_km"]].to_numpy().T
            df_dtec["lts"], df_dtec["lns"], df_dtec["els"] = ipp_calc2(
                (x, y, z), df_stations.loc[station]
            )
            df_dtec = df_dtec.drop(columns=["x_km", "y_km", "z_km"])
            df_dtec = df_dtec.dropna(subset=["lts", "lns", "els"])

            if df_dtec.empty:
                n_empty += 1
                log.info("[%d/%d %5.1f%%] %d/%03d %s sem IPP válido",
                         i, total, pct, year, doy, station)
                continue

            # Escreve em .tmp e renomeia: nunca fica um parquet parcial.
            out_path = _ipp_path(out_dir, station, year, doy)
            tmp_path = out_path.with_suffix(".parquet.tmp")
            df_dtec.to_parquet(tmp_path)          # mantém 'horario' como índice
            tmp_path.replace(out_path)
            n_done += 1
            log.info("[%d/%d %5.1f%%] %d/%03d IPP OK  %s",
                     i, total, pct, year, doy, station)
        except Exception:
            n_failed += 1
            log.exception("[%d/%d %5.1f%%] IPP FAIL %d/%03d %s",
                          i, total, pct, year, doy, station)

    log.info(
        "%d/%03d IPP: %d gravadas, %d já existiam, %d sem IPP válido, %d falhas (de %d)",
        year, doy, n_done, n_skipped, n_empty, n_failed, total,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    parser = build_arg_parser("IPP a partir do dTEC — todos os pares year/doy")
    parser.add_argument("--overwrite", action="store_true",
                        help="recalcula mesmo as estações já gravadas")
    args = parser.parse_args()

    pairs = sorted({(int(y), int(d)) for y, d in resolve_pairs(args, PAIRS_REFERENCE)})
    log.info("Encontradas %d pastas year/doy para processar", len(pairs))

    n_days = len(pairs)
    for k, (year, doy) in enumerate(pairs, start=1):
        log.info("=== %d / doy %03d  [dia %d/%d, %.1f%%] ===",
                 year, doy, k, n_days, 100.0 * k / n_days)
        try:
            run(year, doy, overwrite=args.overwrite)
        except Exception:
            # Um dia problemático não derruba os demais.
            log.exception("IPP FAIL %d/%03d (seguindo para o próximo dia)", year, doy)


if __name__ == "__main__":
    main()
