# -*- coding: utf-8 -*-
"""Etapa 01 — Download de arquivos (C)RINEX das redes GNSS.

Redes suportadas:

- ``rbmc``   — Rede Brasileira de Monitoramento Contínuo (IBGE).
               Faz *scraping* do diretório FTP público, baixando tudo que está
               disponível para o DOY.
- ``ramsac`` — Red Argentina de Monitoreo Satelital Continuo (IGN).
               Itera pela lista fixa de estações em ``data/stations_ramsac.dat``
               e tenta baixar cada uma.

Saída padrão: ``DB_ROOT/gnss/crinex/{year}/{doy}/``.

Uso::

    python 01_download.py --year 2025 --doy 298 --network rbmc
    python 01_download.py --year 2025 --doy 298 --network ramsac
    python 01_download.py --year 2025 --doy 298 --network both
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from urllib.parse import urljoin

import requests
import urllib3
from bs4 import BeautifulSoup

import shared.config as cfg

log = logging.getLogger("download")

# URLs das redes -----------------------------------------------------------
RBMC_BASE = (
    "http://geoftp.ibge.gov.br/informacoes_sobre_posicionamento_geodesico"
    "/rbmc/dados/{year}/{doy:03d}/"
)
RAMSAC_BASE = "https://ramsac.ign.gob.ar/api/v1/rinex/download/15/"

# Lista de estações RAMSAC (relativa ao diretório do script)
RAMSAC_STATIONS_FILE = Path(__file__).parent / "data" / "stations_ramsac.dat"

VALID_EXTENSIONS = (".zip", ".gz", ".Z")
CHUNK_SIZE = 8192
REQUEST_TIMEOUT = 60  # segundos


# =========================================================================
# Helpers
# =========================================================================

def download_stream(url: str, dest: Path, verify_ssl: bool = True) -> bool:
    """Baixa ``url`` para ``dest`` via streaming. Retorna True em sucesso.

    Em falha, remove o arquivo parcial se tiver sido criado.
    """
    try:
        with requests.get(
            url, stream=True, timeout=REQUEST_TIMEOUT, verify=verify_ssl
        ) as r:
            if r.status_code != 200:
                log.debug("%s → HTTP %d", url, r.status_code)
                return False
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
        return True
    except requests.RequestException as exc:
        log.warning("Falha em %s: %s", url, exc)
        dest.unlink(missing_ok=True)
        return False


# =========================================================================
# RBMC / IBGE
# =========================================================================

def list_rbmc_files(url: str) -> list[str]:
    """Retorna os hrefs de arquivos compactados no listing HTML do FTP."""
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return [
        link.get("href")
        for link in soup.find_all("a")
        if link.get("href") and link.get("href").endswith(VALID_EXTENSIONS)
    ]


def download_rbmc(year: int, doy: int, out_dir: Path) -> None:
    base_url = RBMC_BASE.format(year=year, doy=doy)
    log.info("[RBMC] Listando %s", base_url)

    hrefs = list_rbmc_files(base_url)
    log.info("[RBMC] %d arquivos encontrados", len(hrefs))

    for i, href in enumerate(hrefs, start=1):
        filename = Path(href).name
        dest = out_dir / filename

        if dest.exists():
            log.info("[RBMC %d/%d] SKIP %s", i, len(hrefs), filename)
            continue

        log.info("[RBMC %d/%d] GET  %s", i, len(hrefs), filename)
        download_stream(urljoin(base_url, href), dest)


# =========================================================================
# RAMSAC / IGN
# =========================================================================

def load_ramsac_stations(path: Path) -> list[str]:
    """Lê o arquivo de estações (strings de 4 caracteres concatenadas)."""
    content = path.read_text().strip().lower()
    return [content[i : i + 4] for i in range(0, len(content), 4)]


def ramsac_filename(station: str, year: int, doy: int) -> str:
    """Gera o nome do arquivo CRINEX do dia para uma estação RAMSAC.

    Padrão RINEX 2: ``{station}{doy:03d}0.{yy}d.Z``  (ex.: ``igm12980.25d.Z``).
    """
    yy = year % 100
    return f"{station}{doy:03d}0.{yy:02d}d.Z"


def download_ramsac(year: int, doy: int, out_dir: Path) -> None:
    if not RAMSAC_STATIONS_FILE.exists():
        log.error("Arquivo de estações RAMSAC não encontrado: %s", RAMSAC_STATIONS_FILE)
        return

    stations = load_ramsac_stations(RAMSAC_STATIONS_FILE)
    log.info("[RAMSAC] %d estações na lista", len(stations))

    # O servidor do IGN costuma ter problema de cadeia SSL; silencia o warning.
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

    total = len(stations)
    ok = 0
    for i, station in enumerate(stations, start=1):
        filename = ramsac_filename(station, year, doy)
        dest = out_dir / filename

        if dest.exists():
            log.info("[RAMSAC %d/%d] SKIP %s", i, total, filename)
            ok += 1
            continue

        url = RAMSAC_BASE + filename
        log.info("[RAMSAC %d/%d] GET  %s", i, total, filename)
        if download_stream(url, dest, verify_ssl=False):
            ok += 1

    log.info("[RAMSAC] %d/%d arquivos baixados ou já existentes", ok, total)


# =========================================================================
# Entrypoint
# =========================================================================

NETWORKS = {
    "rbmc": download_rbmc,
    "ramsac": download_ramsac,
}


def run(year: int, doy: int, network: str = "both") -> None:
    out_dir = cfg.crinex_dir(year, doy)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = list(NETWORKS) if network == "both" else [network]
    for net in targets:
        NETWORKS[net](year, doy, out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download (C)RINEX das redes GNSS")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--doy", type=int, required=True, help="Dia do ano (1..366)")
    p.add_argument(
        "--network",
        choices=["rbmc", "ramsac", "both"],
        default="both",
        help="Qual rede baixar (padrão: both).",
    )
    return p.parse_args()


if __name__ == "__main__":
    cfg.setup_logging()
    args = parse_args()
    run(args.year, args.doy, args.network)
