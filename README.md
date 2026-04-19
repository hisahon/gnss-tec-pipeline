# gnss-tec-pipeline

Pipeline em Python para processamento de dados GNSS visando o cálculo de
**TEC** (Total Electron Content) ionosférico, IPP (Ionospheric Piercing Point)
e posições de satélite.

Redes suportadas atualmente:

- **RBMC** — Rede Brasileira de Monitoramento Contínuo (IBGE)
- **RAMSAC** — Red Argentina de Monitoreo Satelital Continuo (IGN)

## Estrutura do pipeline

```
[01_download.py]   Baixa (C)RINEX de RBMC (scraping FTP) e/ou RAMSAC (lista fixa)
       │
       ▼
[02_extract.py]    Descompacta .zip/.gz/.Z e converte CRINEX → RINEX (hatanaka)
       │
       ▼
[03_process.py]    Lê RINEX, calcula TEC (prange/pcarrier), IPP e posições
                   dos satélites; salva Parquet por PRN.
```

Executar tudo:

```bash
python run_pipeline.py --year 2025 --doy 298
python run_pipeline.py --year 2025 --doy 298 --network rbmc     # só Brasil
python run_pipeline.py --year 2025 --doy 298 --network ramsac   # só Argentina
```

Etapa por etapa:

```bash
python 01_download.py --year 2025 --doy 298 --network both
python 02_extract.py  --year 2025 --doy 298
python 03_process.py  --year 2025 --doy 298
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Configuração

Copie `.env.example` para `.env` e ajuste os caminhos da sua máquina:

```bash
cp .env.example .env
```

Conteúdo:

```
DB_ROOT=/caminho/para/seu/banco_de_dados
```

O pipeline cria, dentro de `DB_ROOT`, a seguinte árvore:

```
DB_ROOT/
├── gnss/
│   ├── crinex/{year}/{doy}/        # arquivos baixados (.zip/.gz/.Z)
│   ├── rinex/{year}/{doy}/         # RINEX decodificado
│   ├── ephemeris/{year}/{doy}/     # SP3 consolidado
│   └── station_positions/          # metadados das estações
└── ionosphere/
    ├── tec/{year}/{doy}/           # pickles intermediários
    ├── tec_pc/{year}/{doy}/        # Parquet por PRN
    └── tec_error/{year}/{doy}/     # arquivos que falharam no processo
```

## Estrutura do código

```
├── config.py                  # paths + logging (substitui o antigo comv.py)
├── rinex.py                   # parser de RINEX v2/v3
├── sp3.py                     # leitura de SP3, interpolação de órbitas, IPP
├── tec.py                     # cálculo de TEC (pseudo-range e carrier phase)
├── 01_download.py             # etapa 1 (rbmc / ramsac / both)
├── 02_extract.py              # etapa 2
├── 03_process.py              # etapa 3
├── run_pipeline.py            # orquestrador
└── data/
    └── stations_ramsac.dat    # lista de estações do IGN
```

## Adicionando novas redes

Para incluir uma terceira rede (por exemplo, CSN do Chile ou SIRGAS-CON),
adicione uma função `download_<rede>(year, doy, out_dir)` em `01_download.py`
e registre-a no dicionário `NETWORKS`. A nova rede passa automaticamente
a aparecer em `--network`.

## Licença

TBD.
