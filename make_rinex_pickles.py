# -*- coding: utf-8 -*-
"""Etapa 3.1 — Conversão RINEX → pickle.

Lê todos os arquivos ``.??o`` em ``DB_ROOT/gnss/rinex/{year}/{doy}`` e
salva um ``.pkl`` correspondente em ``tec_dir(year, doy)``. Pickles que
já existem são pulados (idempotente).

Arquivos RINEX que não puderem ser processados são MOVIDOS para
``DB_ROOT/gnss/quarantine/{year}/{doy}`` (o original é removido da pasta
de origem).

Depois de corrigir o problema de um arquivo em quarentena, basta movê-lo de
volta para ``DB_ROOT/gnss/rinex/{year}/{doy}`` e rodar o script normalmente.

Uso (terminal)::

    python make_rinex_pickles.py                       # toda a árvore
    python make_rinex_pickles.py --year 2025           # ano inteiro
    python make_rinex_pickles.py --year 2025 --doy 298 # um único dia

Modo debug (apenas Spyder)
--------------------------
Serve para investigar por que os arquivos da quarentena falham.

1. Ajuste a constante ``DEBUG_QUARANTINE = True`` no topo do arquivo.
2. Rode o script dentro do Spyder (console IPython do Spyder). Os filtros
   ``--year`` / ``--doy`` continuam valendo, e a quarentena de cada dia é
   a de ``DB_ROOT/gnss/quarantine/{year}/{doy}``.
3. O script lê os ``.??o`` direto da quarentena, sem gravar pickles e sem
   mover arquivos. A exceção NÃO é capturada: ela sobe no primeiro arquivo
   que falhar. Use ``%debug`` no console para inspecionar o ponto da falha
   (ou rode com o depurador do Spyder).
4. Volte ``DEBUG_QUARANTINE = False`` ao terminar.

No terminal (ou em "Execute in an external system terminal" do Spyder) a
constante é ignorada, com um aviso no log, e o processamento normal roda.
"""

from __future__ import annotations

import logging
import pickle
import shutil
import sys
from pathlib import Path

from shared.config import rinex_dir, tec_dir, setup_logging
import shared.rinex as rnx

from shared.pipeline_utils import build_arg_parser, resolve_pairs
from shared.prn_store import list_files

log = logging.getLogger("process.rinex_to_pickle")

# Ligue manualmente (True) para depurar os RINEX que estão na quarentena.
# Só é respeitado dentro do Spyder; no terminal é ignorado.
DEBUG_QUARANTINE = False


def running_in_spyder() -> bool:
    """True apenas no console/kernel do Spyder (não no terminal)."""
    return "spyder_kernels" in sys.modules


# --------------------------------------------------------------------------
# Núcleo
# --------------------------------------------------------------------------

def quarantine_dir_for(rinex_folder: Path) -> Path:
    """Pasta de quarentena espelhando a estrutura ``{year}/{doy}`` do RINEX.

    ``DB_ROOT/gnss/rinex/{year}/{doy}`` → ``DB_ROOT/gnss/quarantine/{year}/{doy}``
    """
    return rinex_folder.parents[2] / "quarantine" / rinex_folder.parent.name / rinex_folder.name


def quarantine_file(rinex_path: Path, quarantine_folder: Path) -> None:
    """Move o RINEX problemático para a quarentena sem interromper o lote."""
    try:
        quarantine_folder.mkdir(parents=True, exist_ok=True)
        dest = quarantine_folder / rinex_path.name
        shutil.move(str(rinex_path), str(dest))
        log.warning("QUARENTENA %s -> %s", rinex_path.name, dest)
    except Exception:
        log.exception("Não foi possível mover %s para a quarentena", rinex_path.name)


def debug_quarantine(quarantine_folder: Path) -> None:
    """Lê os RINEX direto da quarentena, sem efeitos colaterais.

    Não grava pickles e não move arquivos. A exceção NÃO é capturada: ela
    sobe no primeiro arquivo que falhar, para permitir depuração (por
    exemplo ``%debug`` no console do Spyder).
    """
    if not quarantine_folder.exists():
        log.warning("DEBUG: pasta de quarentena inexistente: %s", quarantine_folder)
        return

    files = list_files(quarantine_folder, "o")
    total = len(files)
    log.info("DEBUG: %d arquivo(s) em %s", total, quarantine_folder)

    for i, rinex_path in enumerate(files, start=1):
        log.info("DEBUG [%d/%d] lendo %s", i, total, rinex_path.name)
        rnx.read_rinex(str(rinex_path))
        log.info("DEBUG [%d/%d] OK %s (leu sem erro)", i, total, rinex_path.name)


def process_rinex_to_pickle(
    rinex_folder: Path, tec_folder: Path, quarantine_folder: Path
) -> None:
    tec_folder.mkdir(parents=True, exist_ok=True)
    files = list_files(rinex_folder, "o")
    total = len(files)

    for i, rinex_path in enumerate(files, start=1):
        out_path = tec_folder / f"{rinex_path.stem}.pkl"
        if out_path.exists():
            log.debug("[%d/%d] SKIP %s (pickle já existe)", i, total, rinex_path.name)
            continue

        try:
            dic = rnx.read_rinex(str(rinex_path))
            with open(out_path, "wb") as f:
                pickle.dump(dic, f)
            log.info("[%d/%d] OK   %s", i, total, rinex_path.name)
        except Exception:
            log.exception("[%d/%d] FAIL %s", i, total, rinex_path.name)
            quarantine_file(rinex_path, quarantine_folder)


def run(year: int, doy: int, debug: bool = False) -> None:
    rinex_folder = rinex_dir(year, doy)
    tec_folder = tec_dir(year, doy)

    lg = logging.LoggerAdapter(log, {"year": year, "doy": doy})

    if debug:
        lg.info("=== 3.1 DEBUG (quarentena) ===")
        debug_quarantine(quarantine_dir_for(rinex_folder))
        return

    if not rinex_folder.exists():
        lg.warning("Pasta de entrada inexistente: %s", rinex_folder)
        return

    lg.info("=== 3.1 RINEX → pickle ===")
    process_rinex_to_pickle(rinex_folder, tec_folder, quarantine_dir_for(rinex_folder))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    args = build_arg_parser(
        "Etapa 3.1 — Conversão RINEX → pickle"
    ).parse_args()

    debug = DEBUG_QUARANTINE and running_in_spyder()
    if DEBUG_QUARANTINE and not debug:
        log.warning("DEBUG_QUARANTINE ignorado: só funciona dentro do Spyder")

    pairs = resolve_pairs(args, 'rinexRAWReferenceFolder')
    log.info("Encontradas %d pastas year/doy para processar", len(pairs))
    for year, doy in pairs:
        log.info("=== %d / doy %03d ===", year, doy)
        run(year, doy, debug=debug)