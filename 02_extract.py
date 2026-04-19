# -*- coding: utf-8 -*-
"""Etapa 02 — Extração + conversão CRINEX → RINEX.

Para cada ``.zip``/``.gz`` em ``DB_ROOT/gnss/crinex/{year}/{doy}``:

1. Descompacta em memória.
2. Converte o arquivo ``.??d`` (CRINEX Hatanaka) para ``.??o`` (RINEX).
3. Grava em ``DB_ROOT/gnss/rinex/{year}/{doy}``.

Uso::

    python 02_extract.py --year 2025 --doy 298
"""

from __future__ import annotations

import argparse
import gzip
import logging
import zipfile
from pathlib import Path

import hatanaka

import config

log = logging.getLogger("extract")


def extract_archive(archive_path: Path) -> dict[str, bytes]:
    """Extrai um .zip ou .gz para memória.

    Retorna um ``dict`` ``{nome_interno: bytes}``. Formatos não suportados
    (como ``.Z``) retornam dicionário vazio com aviso.
    """
    suffix = archive_path.suffix.lower()
    out: dict[str, bytes] = {}

    if suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            for name in zf.namelist():
                with zf.open(name) as f:
                    out[name] = f.read()

    elif suffix == ".gz":
        with gzip.open(archive_path, "rb") as gz:
            out[archive_path.stem] = gz.read()

    else:
        log.warning("Formato não suportado: %s", archive_path.name)

    return out


def convert_crx_to_rnx(files: dict[str, bytes]) -> dict[str, bytes]:
    """Converte o arquivo CRINEX (``*.??d``) presente no dict para RINEX (``*.??o``).

    O arquivo CRINEX original é removido do dict após a conversão.
    """
    crx_key = next((k for k in files if k.lower().endswith("d")), None)
    if crx_key is None:
        return files

    try:
        rnx_data = hatanaka.crx2rnx(files[crx_key])
    except Exception as exc:
        log.error("Falha na conversão de %s: %s", crx_key, exc)
        return files

    new_key = crx_key[:-1] + "o"
    files[new_key] = rnx_data.encode() if isinstance(rnx_data, str) else rnx_data
    del files[crx_key]
    log.debug("Convertido %s → %s", crx_key, new_key)
    return files


def save_files(files: dict[str, bytes], out_dir: Path) -> None:
    """Salva em disco apenas arquivos que NÃO terminam em 'd' (ou seja,
    já convertidos / não-CRINEX, como o RINEX de navegação .??n)."""
    for name, content in files.items():
        if name.endswith("d"):
            continue
        dest = out_dir / name
        if isinstance(content, str):
            content = content.encode("ascii", errors="ignore")
        dest.write_bytes(content)
        log.info("Gravado %s", dest.name)


def run(year: int, doy: int) -> None:
    in_dir = config.crinex_dir(year, doy)
    out_dir = config.rinex_dir(year, doy)
    out_dir.mkdir(parents=True, exist_ok=True)

    archives = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in {".zip", ".gz"})
    log.info("%d arquivos para processar em %s", len(archives), in_dir)

    for i, archive in enumerate(archives, start=1):
        log.info("[%d/%d] %s", i, len(archives), archive.name)
        try:
            files = extract_archive(archive)
            files = convert_crx_to_rnx(files)
            save_files(files, out_dir)
        except Exception:
            log.exception("Erro ao processar %s", archive.name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extração + conversão CRINEX→RINEX")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--doy", type=int, required=True)
    return p.parse_args()


if __name__ == "__main__":
    config.setup_logging()
    args = parse_args()
    run(args.year, args.doy)
