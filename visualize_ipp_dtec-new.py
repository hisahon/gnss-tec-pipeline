# -*- coding: utf-8 -*-
"""Helpers de LEITURA/ANÁLISE sobre o layout long particionado de TEC.

Layout assumido::

    tec_pc/
      {year}/{doy:03d}/
        G/{station}.parquet     ← prn, horario, tec_pcarrier
        R/{station}.parquet
        E/{station}.parquet
        C/{station}.parquet

Funções principais
------------------
- ``load_raw``       : lê 1+ dias de uma estação (ou de todas), formato long.
- ``load_smooth``    : igual ao load_raw, mas com média móvel por PRN.
                       Lê automaticamente vizinhos (DOY±1) pra cobrir bordas.
- ``pivot_wide``     : reconstrói o formato "1 coluna por estação" sob demanda
                       (substitui o parquet wide antigo, em memória).
- ``available_days`` : lista (year, doy) presentes no dataset.
- ``available_stations`` : lista estações de um (year, doy[, const]).

Tudo via DuckDB — você pode rodar as mesmas queries no terminal::

    duckdb -c "
      SELECT prn, COUNT(*) FROM read_parquet('tec_pc/2025/298/G/*.parquet')
      GROUP BY prn ORDER BY prn
    "
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


# --------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------

def _doy_dir(root: Path, year: int, doy: int) -> Path:
    return root / f"{year}" / f"{doy:03d}"


def _station_files(
    root: Path,
    year: int,
    doy: int,
    station: str | None = None,
    constellations: Iterable[str] | None = None,
) -> list[Path]:
    """
    Resolve os parquets para um (year, doy), opcionalmente filtrando por
    estação e/ou constelações. Constelações default = G/R/E/C.
    """
    base = _doy_dir(root, year, doy)
    if not base.exists():
        return []

    consts = list(constellations) if constellations else ["G", "R", "E", "C"]
    pattern = (station.lower() if station else "*") + ".parquet"

    out: list[Path] = []
    for c in consts:
        out.extend((base / c).glob(pattern))
    return out


def _doy_neighbors(year: int, doy: int) -> list[tuple[int, int]]:
    """
    Devolve [(year, doy-1), (year, doy), (year, doy+1)] tratando bordas de
    ano (incluindo ano bissexto). DOYs inválidos são descartados.
    """
    base = dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)
    return [
        ((base + dt.timedelta(days=k)).year,
         (base + dt.timedelta(days=k)).timetuple().tm_yday)
        for k in (-1, 0, 1)
    ]


# --------------------------------------------------------------------------
# Inventário do dataset
# --------------------------------------------------------------------------

def available_days(root: Path) -> list[tuple[int, int]]:
    """Devolve lista ordenada de (year, doy) presentes em disco."""
    out: list[tuple[int, int]] = []
    if not root.exists():
        return out
    for y_dir in sorted(root.iterdir()):
        if not y_dir.is_dir() or not y_dir.name.isdigit():
            continue
        for d_dir in sorted(y_dir.iterdir()):
            if d_dir.is_dir() and d_dir.name.isdigit():
                out.append((int(y_dir.name), int(d_dir.name)))
    return out


def available_stations(
    root: Path,
    year: int,
    doy: int,
    constellation: str | None = None,
) -> list[str]:
    """Lista estações presentes em um (year, doy[, constellation])."""
    base = _doy_dir(root, year, doy)
    if not base.exists():
        return []
    consts = [constellation] if constellation else ["G", "R", "E", "C"]
    stations: set[str] = set()
    for c in consts:
        for p in (base / c).glob("*.parquet"):
            stations.add(p.stem)
    return sorted(stations)


# --------------------------------------------------------------------------
# Leitura raw
# --------------------------------------------------------------------------

def load_raw(
    root: Path,
    year: int,
    doy: int | Iterable[int],
    station: str | None = None,
    constellations: Iterable[str] | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """
    Lê o TEC raw em formato long.

    Parâmetros
    ----------
    root : raiz do dataset (ex.: tec_pc/)
    year : ano (assume mesmo ano para todos os doys passados)
    doy  : 1 doy (int) ou vários (lista). Para cruzar fronteiras de ano,
           passe uma lista com (year, doy) e use `load_raw_pairs` (abaixo).
    station : se None, lê todas as estações.
    constellations : ex. ["G"]; default = todas.

    Retorno
    -------
    DataFrame long: prn, horario, tec_pcarrier, station[, year, doy]
    """
    doys = [doy] if isinstance(doy, int) else list(doy)
    pairs = [(year, d) for d in doys]
    return load_raw_pairs(root, pairs, station, constellations, con)


def load_raw_pairs(
    root: Path,
    pairs: list[tuple[int, int]],
    station: str | None = None,
    constellations: Iterable[str] | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """
    Versão que aceita lista explícita de (year, doy) — útil pra cruzar
    fronteira de ano (ex.: 31/dez + 01/jan).
    """
    files: list[Path] = []
    for y, d in pairs:
        files.extend(_station_files(root, y, d, station, constellations))

    if not files:
        return pd.DataFrame(columns=["prn", "horario", "tec_pcarrier", "station"])

    own_con = con is None
    con = con or duckdb.connect(":memory:")
    try:
        # filename=true injeta o caminho completo; extraímos station do basename
        df = con.execute(f"""
            SELECT
              prn,
              horario,
              tec_pcarrier,
              regexp_extract(filename, '([^/\\\\]+)\\.parquet$', 1) AS station
            FROM read_parquet({[str(p) for p in files]!r}, filename=true)
            ORDER BY station, prn, horario
        """).df()
        return df
    finally:
        if own_con:
            con.close()


# --------------------------------------------------------------------------
# Leitura com suavização (janela rolante cruzando dias)
# --------------------------------------------------------------------------

def load_smooth(
    root: Path,
    year: int,
    doy: int,
    station: str,
    constellations: Iterable[str] | None = None,
    window_hours: int = 24,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """
    Lê uma estação no DOY central + vizinhos (DOY±1), aplica média móvel
    centrada por PRN com janela `window_hours`, e devolve SÓ o dia central.

    A janela é simétrica: ±(window_hours/2) em torno de cada timestamp.
    Cruza fronteira de dia naturalmente porque os 3 dias são lidos como
    uma série contínua.

    Inclui coluna ``n_obs`` com o nº de pontos usados na média (útil pra
    descartar bordas onde não há vizinhos suficientes — embora aqui só
    devolvamos o dia central, onde esse problema não existe).
    """
    pairs = _doy_neighbors(year, doy)
    files: list[Path] = []
    for y, d in pairs:
        files.extend(_station_files(root, y, d, station, constellations))

    if not files:
        return pd.DataFrame(
            columns=["prn", "horario", "tec_pcarrier", "tec_smooth", "n_obs"]
        )

    half = window_hours // 2

    # Limites do dia central [day_start, day_end)
    day_start = dt.datetime(year, 1, 1) + dt.timedelta(days=doy - 1)
    day_end = day_start + dt.timedelta(days=1)

    own_con = con is None
    con = con or duckdb.connect(":memory:")
    try:
        df = con.execute(f"""
            WITH src AS (
              SELECT prn, horario, tec_pcarrier
              FROM read_parquet({[str(p) for p in files]!r})
            ),
            windowed AS (
              SELECT
                prn,
                horario,
                tec_pcarrier,
                AVG(tec_pcarrier) OVER w  AS tec_smooth,
                COUNT(*)          OVER w  AS n_obs
              FROM src
              WINDOW w AS (
                PARTITION BY prn
                ORDER BY horario
                RANGE BETWEEN INTERVAL {half} HOUR PRECEDING
                          AND INTERVAL {half} HOUR FOLLOWING
              )
            )
            SELECT *
            FROM windowed
            WHERE horario >= TIMESTAMP '{day_start.isoformat()}'
              AND horario <  TIMESTAMP '{day_end.isoformat()}'
            ORDER BY prn, horario
        """).df()
        return df
    finally:
        if own_con:
            con.close()


# --------------------------------------------------------------------------
# Reconstrução wide (substitui o parquet "1 coluna por estação" antigo)
# --------------------------------------------------------------------------

def pivot_wide(
    root: Path,
    year: int,
    doy: int,
    prn: str,
    constellation: str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """
    Reconstrói, em memória, o formato wide antigo: 1 coluna por estação,
    para um PRN específico em um DOY. Equivale ao parquet wide do layout
    anterior, mas calculado sob demanda — sem custo de storage.

    Parâmetros
    ----------
    prn : ex. 'G05'. A constelação é deduzida do 1º caractere se não passada.
    """
    const = constellation or prn[0]
    files = _station_files(root, year, doy, station=None, constellations=[const])
    if not files:
        return pd.DataFrame(columns=["horario"])

    own_con = con is None
    con = con or duckdb.connect(":memory:")
    try:
        df = con.execute(f"""
            PIVOT (
              SELECT
                horario,
                regexp_extract(filename, '([^/\\\\]+)\\.parquet$', 1) AS station,
                tec_pcarrier
              FROM read_parquet({[str(p) for p in files]!r}, filename=true)
              WHERE prn = '{prn}'
            )
            ON station USING first(tec_pcarrier)
            GROUP BY horario
            ORDER BY horario
        """).df()
        return df
    finally:
        if own_con:
            con.close()


# --------------------------------------------------------------------------
# Exemplo de uso (rode como script)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Helpers de leitura TEC long")
    ap.add_argument("root", type=Path, help="raiz do dataset (tec_pc/)")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--doy", type=int, required=True)
    ap.add_argument("--station", type=str, default=None)
    ap.add_argument("--prn", type=str, default=None)
    ap.add_argument(
        "--mode",
        choices=["raw", "smooth", "wide", "inventory"],
        default="inventory",
    )
    ap.add_argument("--window", type=int, default=24, help="janela em horas (smooth)")
    args = ap.parse_args()

    if args.mode == "inventory":
        days = available_days(args.root)
        print(f"{len(days)} dias presentes")
        if days:
            print("primeiro:", days[0], "último:", days[-1])
        sts = available_stations(args.root, args.year, args.doy)
        print(f"{len(sts)} estações em {args.year}/{args.doy:03d}")

    elif args.mode == "raw":
        df = load_raw(args.root, args.year, args.doy, args.station)
        print(df)
        print(df.shape)

    elif args.mode == "smooth":
        if not args.station:
            ap.error("--station é obrigatório no modo smooth")
        df = load_smooth(
            args.root, args.year, args.doy, args.station, window_hours=args.window
        )
        print(df)
        print(df.shape)

    elif args.mode == "wide":
        if not args.prn:
            ap.error("--prn é obrigatório no modo wide")
        df = pivot_wide(args.root, args.year, args.doy, args.prn)
        print(df)
        print(df.shape)
