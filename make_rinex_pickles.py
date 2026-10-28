# -*- coding: utf-8 -*-
"""Step 3.1 — RINEX → pickle conversion.

Reads every ``.??o`` file in ``DB_ROOT/gnss/rinex/{year}/{doy}`` and saves a
matching ``.pkl`` in ``tec_dir(year, doy)``. Pickles that already exist are
skipped (idempotent).

RINEX files that cannot be processed are MOVED to
``DB_ROOT/gnss/quarantine/{year}/{doy}`` (the original is removed from the
source folder).

After fixing the problem with a quarantined file, just move it back to
``DB_ROOT/gnss/rinex/{year}/{doy}`` and run the script normally.

Usage (terminal)::

    python make_rinex_pickles.py                       # whole tree
    python make_rinex_pickles.py --year 2025           # whole year
    python make_rinex_pickles.py --year 2025 --doy 298 # a single day
    python make_rinex_pickles.py --workers 4           # whole tree, 4 processes

Parallel mode (terminal only)
-----------------------------
``--workers N`` distributes the year/doy folders across N worker processes
(capped at ``MAX_WORKERS`` = 6), one folder per task. Without the flag the
processing is serial, exactly as before. The flag is ignored (with a warning)
inside Spyder, and there is nothing to distribute when only one folder is
selected. Log lines from different days may interleave.

Debug mode (Spyder only)
------------------------
Use it to investigate why the quarantined files fail.

1. Set the constant ``DEBUG_QUARANTINE = True`` at the top of the file.
2. Run the script inside Spyder (Spyder's IPython console). The ``--year`` /
   ``--doy`` filters still apply, and each day's quarantine folder is
   ``DB_ROOT/gnss/quarantine/{year}/{doy}``.
3. The script reads the ``.??o`` files straight from quarantine, without
   writing pickles and without moving files. The exception is NOT caught: it
   is raised on the first file that fails. Use ``%debug`` in the console to
   inspect the point of failure (or run with Spyder's debugger).
4. Set ``DEBUG_QUARANTINE = False`` again when you are done.

In the terminal (or with Spyder's "Execute in an external system terminal")
the constant is ignored, with a warning in the log, and normal processing
runs.
"""

from __future__ import annotations

import logging
import pickle
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from shared.config import rinex_dir, tec_dir, setup_logging
import shared.rinex as rnx

from shared.pipeline_utils import build_arg_parser, resolve_pairs
from shared.prn_store import list_files

log = logging.getLogger("process.rinex_to_pickle")

# Set to True manually to debug the RINEX files sitting in quarantine.
# Only honored inside Spyder; ignored in the terminal.
DEBUG_QUARANTINE = False

# Upper limit for ``--workers`` (terminal only).
MAX_WORKERS = 6


def running_in_spyder() -> bool:
    """True only in the Spyder console/kernel (not in the terminal)."""
    return "spyder_kernels" in sys.modules


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------

def date_str(year: int, doy: int) -> str:
    """Convert year/doy to ``YYYY-MM-DD`` (``?`` if the doy is invalid)."""
    try:
        d = datetime.strptime(f"{year}-{doy:03d}", "%Y-%j")
    except ValueError:
        return "?"
    # strptime accepts doy 366 in a non-leap year and rolls over to the next year
    return d.strftime("%Y-%m-%d") if d.year == year else "?"


def quarantine_dir_for(rinex_folder: Path) -> Path:
    """Quarantine folder mirroring the RINEX ``{year}/{doy}`` structure.

    ``DB_ROOT/gnss/rinex/{year}/{doy}`` → ``DB_ROOT/gnss/quarantine/{year}/{doy}``
    """
    return rinex_folder.parents[2] / "quarantine" / rinex_folder.parent.name / rinex_folder.name


def quarantine_file(rinex_path: Path, quarantine_folder: Path) -> None:
    """Move the problematic RINEX to quarantine without interrupting the batch."""
    try:
        quarantine_folder.mkdir(parents=True, exist_ok=True)
        dest = quarantine_folder / rinex_path.name
        shutil.move(str(rinex_path), str(dest))
        log.warning("QUARANTINE %s -> %s", rinex_path.name, dest)
    except Exception:
        log.exception("Could not move %s to quarantine", rinex_path.name)


def debug_quarantine(quarantine_folder: Path) -> None:
    """Read the RINEX files straight from quarantine, with no side effects.

    Does not write pickles and does not move files. The exception is NOT
    caught: it is raised on the first file that fails, so it can be debugged
    (for example with ``%debug`` in the Spyder console).
    """
    if not quarantine_folder.exists():
        log.warning("DEBUG: quarantine folder does not exist: %s", quarantine_folder)
        return

    files = list_files(quarantine_folder, "o")
    total = len(files)
    log.info("DEBUG: %d file(s) in %s", total, quarantine_folder)

    for i, rinex_path in enumerate(files, start=1):
        log.info("DEBUG [%d/%d] reading %s", i, total, rinex_path.name)
        rnx.read_rinex(str(rinex_path))
        log.info("DEBUG [%d/%d] OK %s (read without errors)", i, total, rinex_path.name)


def process_rinex_to_pickle(
    rinex_folder: Path, tec_folder: Path, quarantine_folder: Path
) -> None:
    tec_folder.mkdir(parents=True, exist_ok=True)
    files = list_files(rinex_folder, "o")
    total = len(files)

    for i, rinex_path in enumerate(files, start=1):
        out_path = tec_folder / f"{rinex_path.stem}.pkl"
        if out_path.exists():
            log.debug("[%d/%d] SKIP %s (pickle already exists)", i, total, rinex_path.name)
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

    quarantine_folder = quarantine_dir_for(rinex_folder)

    lg = logging.LoggerAdapter(log, {"year": year, "doy": doy})

    lg.info(
        "########## %d / doy %03d (%s) | 3.1 %s ##########",
        year, doy, date_str(year, doy),
        "DEBUG (quarantine)" if debug else "RINEX → pickle",
    )
    lg.info("  RINEX:      %s", rinex_folder)
    lg.info("  Pickle:     %s", tec_folder)
    lg.info("  Quarantine: %s", quarantine_folder)

    if debug:
        debug_quarantine(quarantine_folder)
        return

    if not rinex_folder.exists():
        lg.warning("Input folder does not exist: %s", rinex_folder)
        return

    process_rinex_to_pickle(rinex_folder, tec_folder, quarantine_folder)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    parser = build_arg_parser("Step 3.1 — RINEX → pickle conversion")
    parser.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help=f"Process N year/doy folders in parallel (max {MAX_WORKERS}). "
             "Terminal only (ignored inside Spyder). Omit for serial processing.",
    )
    args = parser.parse_args()

    in_spyder = running_in_spyder()

    debug = DEBUG_QUARANTINE and in_spyder
    if DEBUG_QUARANTINE and not debug:
        log.warning("DEBUG_QUARANTINE ignored: it only works inside Spyder")

    workers = args.workers
    if workers and in_spyder:
        log.warning("--workers ignored: parallel mode only works in the terminal")
        workers = None
    elif workers and workers > MAX_WORKERS:
        log.warning("--workers %d capped at %d", workers, MAX_WORKERS)
        workers = MAX_WORKERS

    pairs = resolve_pairs(args, 'rinexRAWReferenceFolder')
    log.info("Found %d year/doy folders to process", len(pairs))

    if workers and workers > 1 and len(pairs) > 1:
        n = min(workers, len(pairs))
        log.info("Parallel mode: %d workers", n)
        with ProcessPoolExecutor(n) as ex:
            futs = {ex.submit(run, y, d): (y, d) for y, d in pairs}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception:
                    log.exception("Day %s failed", futs[fut])
    else:
        for year, doy in pairs:
            run(year, doy, debug=debug)