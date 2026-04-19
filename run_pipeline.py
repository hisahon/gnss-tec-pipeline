# -*- coding: utf-8 -*-
"""Orquestrador do pipeline completo.

Executa, em sequência, as três etapas para um ``(year, doy)`` dado::

    python run_pipeline.py --year 2025 --doy 298
    python run_pipeline.py --year 2025 --doy 298 --network rbmc
    python run_pipeline.py --year 2025 --doy 298 --skip 1

Use ``--skip`` para pular etapas e ``--network`` para escolher a(s) rede(s)
na etapa de download.
"""

from __future__ import annotations

import argparse
import importlib
import logging

import config

log = logging.getLogger("pipeline")


STEPS = {
    1: ("01_download", "Download CRINEX"),
    2: ("02_extract", "Extract + CRINEX→RINEX"),
    3: ("03_process", "TEC + satellite positions"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline GNSS → TEC")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--doy", type=int, required=True)
    p.add_argument(
        "--network",
        choices=["rbmc", "ramsac", "both"],
        default="both",
        help="Rede a usar na etapa 1 (padrão: both).",
    )
    p.add_argument(
        "--skip",
        type=int,
        nargs="*",
        default=[],
        choices=list(STEPS.keys()),
        help="Números de etapas a pular (1, 2, 3).",
    )
    return p.parse_args()


def main() -> None:
    config.setup_logging()
    args = parse_args()

    for step_num, (module_name, label) in STEPS.items():
        if step_num in args.skip:
            log.info("--- Etapa %d (%s) PULADA ---", step_num, label)
            continue

        log.info("--- Etapa %d: %s ---", step_num, label)
        module = importlib.import_module(module_name)

        # A etapa 1 aceita --network; as demais não.
        if step_num == 1:
            module.run(args.year, args.doy, args.network)
        else:
            module.run(args.year, args.doy)

    log.info("Pipeline concluído.")


if __name__ == "__main__":
    main()
