# -*- coding: utf-8 -*-
"""Etapa 3.1 — Conversão RINEX → pickle.

Lê todos os arquivos ``.??o`` em ``DB_ROOT/gnss/rinex/{year}/{doy}`` e
salva um ``.pkl`` correspondente em ``tec_dir(year, doy)``. Pickles que
já existem são pulados (idempotente).

Uso::

    python 03a_rinex_to_pickle.py                       # toda a árvore
    python 03a_rinex_to_pickle.py --year 2025           # ano inteiro
    python 03a_rinex_to_pickle.py --year 2025 --doy 298 # um único dia
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path 

from shared.config import rinex_dir, tec_dir, setup_logging
import shared.rinex as rnx

from shared.pipeline_utils import build_arg_parser, resolve_pairs
from shared.prn_store import list_files

log = logging.getLogger("process.rinex_to_pickle")


# --------------------------------------------------------------------------
# Núcleo
# --------------------------------------------------------------------------

def process_rinex_to_pickle(rinex_folder: Path, tec_folder: Path) -> None:
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


def run(year: int, doy: int) -> None:
    rinex_folder = rinex_dir(year, doy)
    tec_folder = tec_dir(year, doy)

    lg = logging.LoggerAdapter(log, {"year": year, "doy": doy})

    if not rinex_folder.exists():
        lg.warning("Pasta de entrada inexistente: %s", rinex_folder)
        return

    lg.info("=== 3.1 RINEX → pickle ===")
    process_rinex_to_pickle(rinex_folder, tec_folder)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    args = build_arg_parser(
        "Etapa 3.1 — Conversão RINEX → pickle"
    ).parse_args()

    pairs = resolve_pairs(args)
    log.info("Encontradas %d pastas year/doy para processar", len(pairs))
    for year, doy in pairs:
        log.info("=== %d / doy %03d ===", year, doy)
        run(year, doy)
