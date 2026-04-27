# -*- coding: utf-8 -*-
"""Etapa 02 — Extração + conversão CRINEX → RINEX.

Percorre toda a estrutura ``DB_ROOT/gnss/crinex/{year}/{doy}`` e, para cada
``.zip`` / ``.gz`` / ``.Z`` encontrado:

1. Descompacta em memória.
2. Converte o arquivo ``.??d`` (CRINEX Hatanaka) para ``.??o`` (RINEX).
3. Grava em ``DB_ROOT/gnss/rinex/{year}/{doy}``.

Arquivos cujo output final já existe são ignorados (reexecução é segura).

Uso::

    python 02_extract.py                    # processa tudo
    python 02_extract.py --year 2025        # processa apenas um ano
    python 02_extract.py --year 2025 --doy 298   # processa apenas um doy
"""

from __future__ import annotations

import argparse
import gzip
import logging
import zipfile
import unlzw3
from pathlib import Path

import hatanaka

import config

log = logging.getLogger("extract")

ARCHIVE_SUFFIXES = {".zip", ".gz", ".z"}


def _detect_format(data: bytes) -> str:
    """Detecta o formato real a partir dos magic bytes do conteúdo.

    É comum em arquivos RINEX a extensão não bater com o conteúdo
    (ex.: um gzip nomeado como ``.Z``). Esta detecção evita isso.
    """
    if len(data) < 4:
        return "unknown"
    if data[:2] == b"\x1f\x8b":
        return "gzip"
    if data[:2] == b"\x1f\x9d":
        return "compress"  # Unix .Z (LZW)
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        return "zip"
    return "unknown"


def extract_archive(archive_path: Path) -> dict[str, bytes]:
    """Extrai um archive para memória, detectando o formato via magic bytes.

    Retorna um ``dict`` ``{nome_interno: bytes}``. Se o conteúdo não for
    reconhecido, retorna dicionário vazio e emite aviso com os bytes iniciais
    pra diagnóstico.
    """
    out: dict[str, bytes] = {}

    with open(archive_path, "rb") as f:
        raw = f.read()

    fmt = _detect_format(raw)

    if fmt == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            for name in zf.namelist():
                with zf.open(name) as zf_file:
                    out[name] = zf_file.read()

    elif fmt == "gzip":
        out[archive_path.stem] = gzip.decompress(raw)

    elif fmt == "compress":
        out[archive_path.stem] = unlzw3.unlzw(raw)

    else:
        log.warning(
            "Formato não reconhecido em %s (magic=%s, ext=%s)",
            archive_path.name,
            raw[:4].hex() if raw else "vazio",
            archive_path.suffix,
        )

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
    """Grava apenas arquivos que NÃO terminam em 'd' (ou seja, já convertidos
    ou não-CRINEX, como o RINEX de navegação ``.??n``)."""
    for name, content in files.items():
        if name.endswith("d"):
            continue
        dest = out_dir / name
        if isinstance(content, str):
            content = content.encode("ascii", errors="ignore")
        dest.write_bytes(content)
        # log.info("Gravado %s", dest.name)


def predicted_outputs(archive_path: Path) -> list[str]:
    """Nomes esperados dos arquivos de saída para um dado archive.

    - ``.gz`` / ``.Z``: nome deduzido do ``stem``; se terminar em ``d`` vira ``o``.
    - ``.zip``: inspeciona a lista interna (leitura apenas do diretório central).

    Retorna lista vazia se não for possível prever (formato desconhecido ou
    zip corrompido); nesse caso o arquivo é tratado como "não processado"
    e tenta-se processá-lo normalmente.
    """
    suffix = archive_path.suffix.lower()

    if suffix in (".gz", ".z"):
        inner = archive_path.stem
        if inner.lower().endswith("d"):
            return [inner[:-1] + "o"]
        return [inner]

    if suffix == ".zip":
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                names = zf.namelist()
        except Exception as exc:
            log.warning("Não foi possível inspecionar %s: %s", archive_path.name, exc)
            return []
        result = []
        for name in names:
            if name.endswith("/"):
                continue
            if name.lower().endswith("d"):
                result.append(name[:-1] + "o")
            else:
                result.append(name)
        return result

    return []


def all_outputs_exist(archive_path: Path, out_dir: Path) -> bool:
    """True se todos os arquivos de saída previstos já estão em ``out_dir``."""
    expected = predicted_outputs(archive_path)
    if not expected:
        return False
    return all((out_dir / name).exists() for name in expected)


def run(year: int, doy: int) -> None:
    in_dir = config.crinex_dir(year, doy)
    out_dir = config.rinex_dir(year, doy)

    if not in_dir.exists():
        log.warning("Pasta de entrada inexistente: %s", in_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    archives = sorted(
        p for p in in_dir.iterdir() if p.suffix.lower() in ARCHIVE_SUFFIXES
    )
    total = len(archives)
    log.info("%d arquivos em %s", total, in_dir)

    skipped = 0
    for i, archive in enumerate(archives, start=1):
        if all_outputs_exist(archive, out_dir):
            skipped += 1
            log.debug("[%d/%d] %s já processado, pulando", i, total, archive.name)
            continue

        log.info("[%d/%d] %s", i, total, archive.name)
        try:
            files = extract_archive(archive)
            files = convert_crx_to_rnx(files)
            save_files(files, out_dir)
        except Exception:
            log.exception("Erro ao processar %s", archive.name)

    if skipped:
        log.info("  → %d de %d já estavam processados (pulados)", skipped, total)


def _crinex_root() -> Path:
    """Deduz o diretório raiz de ``crinex/`` a partir de ``config.crinex_dir``.

    A estrutura esperada é ``.../crinex/{year}/{doy}``; portanto subir dois
    níveis a partir de qualquer par (year, doy) retorna a raiz.
    """
    return config.crinex_dir(2000, 1).parent.parent


def discover_year_doys(year_filter: int | None = None) -> list[tuple[int, int]]:
    """Descobre todos os pares ``(year, doy)`` existentes sob ``crinex/``.

    Se ``year_filter`` for fornecido, retorna apenas os doys daquele ano.
    """
    root = _crinex_root()
    if not root.exists():
        log.error("Pasta crinex não encontrada: %s", root)
        return []

    pairs: list[tuple[int, int]] = []
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        if year_filter is not None and year != year_filter:
            continue
        for doy_dir in sorted(year_dir.iterdir()):
            if not doy_dir.is_dir() or not doy_dir.name.isdigit():
                continue
            doy = int(doy_dir.name)
            pairs.append((year, doy))
    return pairs


def run_all(year_filter: int | None = None) -> None:
    pairs = discover_year_doys(year_filter)
    log.info("Encontradas %d pastas year/doy para processar", len(pairs))
    for year, doy in pairs:
        log.info("=== %d / doy %03d ===", year, doy)
        run(year, doy)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extração + conversão CRINEX→RINEX (toda a árvore crinex/)"
    )
    p.add_argument("--year", type=int, help="Filtrar por um ano específico (opcional)")
    p.add_argument("--doy", type=int, help="Filtrar por um doy específico (requer --year)")
    return p.parse_args()


if __name__ == "__main__":
    config.setup_logging()
    args = parse_args()

    if args.year is not None and args.doy is not None:
        run(args.year, args.doy)
    elif args.year is not None:
        run_all(year_filter=args.year)
    else:
        run_all()
