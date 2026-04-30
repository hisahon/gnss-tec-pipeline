# -*- coding: utf-8 -*-
"""Etapa 3.c — Download dos arquivos SP3 brutos (efemérides precisas).

Baixa os arquivos SP3 ``Final`` do CDDIS para cada ``(year, doy)``,
seguindo a ordem de prioridade ``COD → WUM → SHA → IAC``. Para no
primeiro provedor que responder com sucesso. Idempotente: se o arquivo
descomprimido já existe em ``ephemeris_dir(year, doy)``, pula.

Pré-requisito: variáveis ``EARTHDATA_USERNAME`` e ``EARTHDATA_PASSWORD``
disponíveis (via ``.env`` ou ambiente).

Uso::

    python 03c_download_sp3.py                       # toda a árvore
    python 03c_download_sp3.py --year 2025           # ano inteiro
    python 03c_download_sp3.py --year 2025 --doy 298 # um único dia
"""

from __future__ import annotations

import logging
from shared.config import ephemeris_dir, doy_to_date, setup_logging

from shared.earthdata_download import download_precise_orbits
from shared.pipeline_utils import build_arg_parser, resolve_pairs

log = logging.getLogger("process.download_sp3")


# --------------------------------------------------------------------------
# Núcleo
# --------------------------------------------------------------------------

def run(year: int, doy: int) -> None:
    """Baixa o SP3 do dia, se ainda não estiver em disco."""
    lg = logging.LoggerAdapter(log, {"year": year, "doy": doy})
    eph_dir = ephemeris_dir(year, doy)
    eph_dir.mkdir(parents=True, exist_ok=True)

    # Skip se já existe algum SP3 (descomprimido) na pasta
    existing = list(eph_dir.glob("*.SP3")) + list(eph_dir.glob("*.sp3"))
    if existing:
        lg.info("SKIP (já existe SP3 em %s)", eph_dir)
        return

    target_date = doy_to_date(year, doy)
    lg.info("=== 3.c Download SP3 ===")
    try:
        download_precise_orbits(target_date)
    except Exception:
        lg.exception("Falha no download do SP3")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    args = build_arg_parser(
        "Etapa 3.c — Download de SP3 brutos do CDDIS"
    ).parse_args()

    pairs = resolve_pairs(args, referencefolder="tecRAWReferenceFolder")
    log.info("Encontradas %d pastas year/doy para processar", len(pairs))
    for year, doy in pairs:
        log.info("=== %d / doy %03d ===", year, doy)
        run(year, doy)
