# gnss-tec-pipeline

Pipeline em Python para processamento de observações GNSS visando o cálculo de
**TEC** (Total Electron Content) ionosférico, coordenadas do **IPP**
(Ionospheric Piercing Point) e posições dos satélites. O projeto atende
principalmente trabalhos de pesquisa em ionosfera sobre a América do Sul, a
partir dos dados de duas redes geodésicas de monitoramento contínuo.

Redes suportadas:

- **RBMC** — Rede Brasileira de Monitoramento Contínuo (IBGE)
- **RAMSAC** — Red Argentina de Monitoreo Satelital Continuo (IGN)

Constelações processadas: **GPS (G)**, **GLONASS (R)**, **Galileo (E)**,
**BeiDou (C)**, **QZSS (J)**, **IRNSS (I)**, **SBAS (S)**.

---

## Sumário

- [Contexto científico](#contexto-científico)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Uso rápido](#uso-rápido)
- [Arquitetura do pipeline](#arquitetura-do-pipeline)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Estrutura dos dados de saída](#estrutura-dos-dados-de-saída)
- [Esquema dos arquivos Parquet](#esquema-dos-arquivos-parquet)
- [Usando os dados (exemplos)](#usando-os-dados-exemplos)
- [Convenções e fórmulas](#convenções-e-fórmulas)
- [Adicionando novas redes](#adicionando-novas-redes)
- [Troubleshooting](#troubleshooting)
- [Referências](#referências)
- [Citação](#citação)
- [Autores e créditos](#autores-e-créditos)
- [Licença](#licença)

---

## Contexto científico

O conteúdo eletrônico total (**TEC**, em unidades TECU = 10¹⁶ elétrons/m²) é
uma grandeza fundamental para descrever o estado da ionosfera e afeta
diretamente a propagação de sinais de rádio, incluindo GNSS. Como a ionosfera
é um meio dispersivo, a diferença de tempo (ou de fase) entre dois sinais
GNSS em frequências distintas permite estimar o TEC ao longo da linha de
visada entre satélite e receptor (slant TEC, **STEC**).

Este pipeline automatiza as três tarefas típicas desse cálculo:

1. **Aquisição** — baixar as observações brutas das redes públicas.
2. **Decodificação** — extrair e converter os arquivos RINEX/CRINEX.
3. **Cálculo** — aplicar as combinações de frequência apropriadas para cada
   constelação (incluindo o caso especial do GLONASS, em FDMA) e interpolar
   as efemérides SP3 para localizar o satélite no instante de observação.

O resultado é um conjunto de arquivos Parquet organizado **por PRN**, com uma
coluna por estação, pronto para análise em pandas/xarray.

---

## Requisitos

- **Python ≥ 3.10** (tipos como `list[str]` e `X | None` no módulo de
  download)
- Sistema operacional: **Linux, macOS ou Windows** (todos os caminhos usam
  `pathlib`; não há dependência de shell)
- **≈ 5 GB de espaço em disco por DOY** se baixar as duas redes e manter
  todos os intermediários (CRINEX + RINEX + pickles + Parquets)
- Conexão estável para a etapa 1 (downloads FTP/HTTPS)

Pacotes Python (ver `requirements.txt`):

| Pacote | Uso |
|--------|-----|
| `numpy`, `pandas`, `pyarrow` | Dataframes e Parquet |
| `scipy` | Interpolação PCHIP das órbitas SP3 |
| `astropy` | Conversões geocêntrico ↔ geodésico (WGS84) |
| `hatanaka` | Descompactação CRINEX → RINEX |
| `requests`, `beautifulsoup4` | Download e scraping do IBGE |
| `python-dotenv` | Leitura do arquivo `.env` |

---

## Instalação

```bash
git clone https://github.com/<seu-usuario>/gnss-tec-pipeline.git
cd gnss-tec-pipeline

python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows PowerShell

pip install -r requirements.txt
```

---

## Configuração

Copie `.env.example` para `.env` e ajuste a raiz do seu banco de dados:

```bash
cp .env.example .env
```

Edite `.env`:

```
DB_ROOT=/caminho/absoluto/para/seu/banco_de_dados
```

O pipeline só escreve dentro de `DB_ROOT` — nenhum arquivo é gravado no
diretório do repositório (exceto os de `data/`, que são versionados).

> **Nota sobre efemérides SP3.** O pipeline espera os arquivos SP3 (órbitas
> precisas) já disponíveis em `DB_ROOT/gnss/ephemeris/{year}/{doy}/`. O
> download dessas efemérides ainda não está automatizado — por enquanto,
> baixe manualmente de um provedor como o CDDIS ou IGS e deixe os `.sp3.gz`
> nessa pasta antes de rodar a etapa 3.

---

## Uso rápido

**Processar um DOY inteiro das duas redes:**

```bash
python run_pipeline.py --year 2025 --doy 298
```

**Apenas uma rede:**

```bash
python run_pipeline.py --year 2025 --doy 298 --network rbmc
python run_pipeline.py --year 2025 --doy 298 --network ramsac
```

**Pular etapas** (útil se já baixou os dados):

```bash
python run_pipeline.py --year 2025 --doy 298 --skip 1       # pula download
python run_pipeline.py --year 2025 --doy 298 --skip 1 2     # só processamento
```

**Etapa por etapa** (maior controle, logs mais focados):

```bash
python 01_download.py --year 2025 --doy 298 --network both
python 02_extract.py  --year 2025 --doy 298
python 03_process.py  --year 2025 --doy 298
```

---

## Arquitetura do pipeline

```
  RBMC (IBGE)              RAMSAC (IGN)
  FTP scraping             lista fixa de 135 estações
       │                          │
       └─────────┬────────────────┘
                 ▼
       ┌──────────────────┐
       │ 01_download.py   │   →  DB_ROOT/gnss/crinex/{year}/{doy}/
       └──────────────────┘      (.zip / .gz / .Z)
                 │
                 ▼
       ┌──────────────────┐
       │ 02_extract.py    │   →  DB_ROOT/gnss/rinex/{year}/{doy}/
       └──────────────────┘      (.??o  RINEX 2  observação)
                 │                (.??n  RINEX 2  navegação — opcional)
                 ▼
       ┌──────────────────┐
       │ 03_process.py    │
       ├──────────────────┤
       │ 3.1 RINEX→pickle │   →  DB_ROOT/ionosphere/tec/{year}/{doy}/*.pkl
       │ 3.2 TEC pcarrier │   →  DB_ROOT/ionosphere/tec_pc/{year}/{doy}/
       │     por PRN      │        TECPC_{year}-{doy}_{PRN}.parquet
       │ 3.3 SP3 → x,y,z  │        (adiciona coordenadas do satélite)
       └──────────────────┘
                 │
                 ▼
       ┌──────────────────────────────────────────────┐
       │ Parquet por PRN: 1 linha por época,          │
       │ 1 coluna por estação + (x_km, y_km, z_km).   │
       │ Estações adicionadas incrementalmente:       │
       │ reexecutar o pipeline preserva os valores    │
       │ já calculados.                               │
       └──────────────────────────────────────────────┘
```

---

## Estrutura do repositório

```
gnss-tec-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── config.py                # Paths + logging; lê DB_ROOT do .env
│
├── rinex.py                 # Parser de RINEX v2 e v3
├── sp3.py                   # Leitura SP3, interpolação (PCHIP), IPP
├── tec.py                   # Cálculo de TEC (prange e pcarrier)
│
├── 01_download.py           # Etapa 1: RBMC (FTP) + RAMSAC (API)
├── 02_extract.py            # Etapa 2: .zip/.gz/.Z → RINEX
├── 03_process.py            # Etapa 3: TEC + posições de satélite
├── run_pipeline.py          # Orquestrador (encadeia 01 → 02 → 03)
│
└── data/
    └── stations_ramsac.dat  # 135 estações do IGN (fixo, versionado)
```

---

## Estrutura dos dados de saída

Dentro de `DB_ROOT`, o pipeline cria e mantém a seguinte árvore:

```
DB_ROOT/
├── gnss/
│   ├── crinex/{year}/{doy}/        # arquivos baixados (.zip/.gz/.Z)
│   ├── rinex/{year}/{doy}/         # RINEX de observação (.??o) e
│   │                               # navegação (.??n), quando disponível
│   ├── ephemeris/{year}/{doy}/     # SP3 (entrada externa) +
│   │                               # {year}{doy}_complete.pkl (cache)
│   └── station_positions/
│       └── SP_{year}-{doy}.parquet # metadados das estações
│
└── ionosphere/
    ├── tec/{year}/{doy}/           # pickles intermediários (após leitura
    │                               # do RINEX, antes da escrita final)
    ├── tec_pc/{year}/{doy}/        # Parquet por PRN (saída principal)
    └── tec_error/{year}/{doy}/     # pickles que falharam; preservados
                                    # para debug posterior
```

---

## Esquema dos arquivos Parquet

### `tec_pc/{year}/{doy}/TECPC_{year}-{doy}_{PRN}.parquet`

Um arquivo por PRN (ex.: `G12`, `R03`, `E21`).

| Coluna     | Tipo            | Descrição                                                |
|------------|-----------------|----------------------------------------------------------|
| `horario`  | datetime64[ns]  | Instante da observação (UTC)                             |
| `{estação}`| float64         | STEC em TECU; uma coluna por estação que observou o PRN  |
| `x_km`     | float64         | Coordenada ECEF do satélite, km (após etapa 3.3)         |
| `y_km`     | float64         | idem                                                      |
| `z_km`     | float64         | idem                                                      |

Exemplo de colunas após processar RBMC e RAMSAC juntos: `horario, alar,
bomj, cord, igm1, ..., x_km, y_km, z_km`.

### `station_positions/SP_{year}-{doy}.parquet`

Uma linha por estação.

| Coluna    | Tipo     | Descrição                                          |
|-----------|----------|----------------------------------------------------|
| `name`    | object   | Código de 4 caracteres                             |
| `x_km`    | float64  | Posição ECEF da estação, km                        |
| `y_km`    | float64  | idem                                               |
| `z_km`    | float64  | idem                                               |
| `ln_deg`  | float64  | Longitude geodésica WGS84, graus                   |
| `lt_deg`  | float64  | Latitude geodésica WGS84, graus                    |
| `h_km`    | float64  | Altitude elipsoidal, km                            |

---

## Usando os dados (exemplos)

**Carregar o STEC de um PRN específico para todas as estações:**

```python
import pandas as pd

df = pd.read_parquet("DB_ROOT/ionosphere/tec_pc/2025/298/TECPC_2025-298_G12.parquet")
print(df.head())
print(f"Estações observando G12: {df.columns.difference(['horario','x_km','y_km','z_km']).tolist()}")
```

**Comparar uma estação brasileira e uma argentina para o mesmo satélite:**

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("DB_ROOT/ionosphere/tec_pc/2025/298/TECPC_2025-298_G12.parquet")

fig, ax = plt.subplots(figsize=(10, 4))
df.plot(x="horario", y=["bomj", "cord"], ax=ax)
ax.set_ylabel("STEC (TECU)")
ax.set_title("G12 — Bom Jesus da Lapa (BR) vs. Córdoba (AR)")
plt.show()
```

**Iterar sobre todos os PRNs de uma constelação:**

```python
from pathlib import Path
import pandas as pd

pq_dir = Path("DB_ROOT/ionosphere/tec_pc/2025/298")
gps_files = sorted(pq_dir.glob("TECPC_*_G??.parquet"))

for f in gps_files:
    df = pd.read_parquet(f)
    print(f.stem, df.shape)
```

**Calcular um dTEC simples (desvio da média móvel de 30 min):**

```python
import pandas as pd

df = pd.read_parquet("DB_ROOT/ionosphere/tec_pc/2025/298/TECPC_2025-298_G12.parquet")
df = df.set_index("horario")

station = "cord"
window = 30 * 60 // 15   # 30 min em amostras de 15 s (ajuste ao seu res)
df["dtec30"] = df[station] - df[station].rolling(window, center=True).mean()
```

---

## Convenções e fórmulas

### Combinação geométrica livre (geometry-free)

Para um satélite observado em duas frequências `f1` e `f2`:

```
STEC = (1 / 40.308) · (f1² · f2²) / (f1² − f2²) · ΔL  [unidades de 10¹⁶ elétrons/m²]
```

- **Pseudo-distância (`prange`):** `ΔL = P1 − P2`, com `P_i` em metros.
- **Fase da portadora (`pcarrier`):** `ΔL = (c/f1)·L1 − (c/f2)·L2`, com
  `L_i` em ciclos e `c = 299 792 458 m/s`.

A constante `40.308` tem unidades de m³/s² e deriva da relação entre índice
de refração ionosférico e densidade eletrônica. A divisão por 10¹⁶ converte
o resultado para TECU.

### Frequências por constelação

Valores extraídos da [tabela oficial RINEX 3.04](https://files.igs.org/pub/data/format/rinex304.pdf)
e embutidos em `tec.gnss_channels()`. GPS L1/L2 = 1575.42 / 1227.60 MHz,
Galileo E1/E5a = 1575.42 / 1176.45 MHz, etc.

**GLONASS** é tratado à parte (`tec.tec_rinex2_glonass`) porque usa FDMA:
cada satélite opera em uma frequência deslocada por um canal `k ∈
[−7, +6]`:

```
f1 = 1602.000 MHz + k · 562.5 kHz
f2 = 1246.000 MHz + k · 437.5 kHz
```

O mapeamento PRN → canal está hardcoded no módulo.

### Interpolação de órbitas

As efemérides SP3 trazem posições de satélite a cada 15 min. Para cada PRN
com observações, o módulo `sp3` constrói um interpolador PCHIP (cúbico,
monotônico) em cada coordenada ECEF e avalia nos instantes das observações
GNSS (`sp3.satellite_coordinates`). Usa `scipy.interpolate.PchipInterpolator`.

### IPP (Ionospheric Piercing Point)

Implementado em `sp3.ipp_calculator` / `sp3.ipp_calculator_numpy`. O modelo
é de **camada fina esférica** com altura ajustável (padrão: 350 km). O
cálculo passa pela conversão das coordenadas do satélite e da estação para
o sistema topocêntrico (Norte-Leste-Vertical), obtenção do ângulo de
elevação e azimute, e finalmente trigonometria esférica para a latitude e
longitude do ponto de perfuração.

---

## Adicionando novas redes

Para incluir uma terceira fonte (por exemplo, CSN do Chile, SIRGAS-CON em
geral, ou uma rede local), edite `01_download.py`:

1. Escreva uma função `download_<rede>(year, doy, out_dir)`.
2. Registre-a no dicionário `NETWORKS` no final do arquivo.
3. A opção aparece automaticamente em `--network`, tanto em
   `01_download.py` quanto em `run_pipeline.py`.

Se a rede oferece um diretório FTP público (como RBMC), reaproveite o
padrão de `list_rbmc_files` + `download_stream`. Se tiver uma lista fixa
de estações (como RAMSAC), coloque o arquivo de estações em `data/` e
siga o padrão de `download_ramsac`.

---

## Troubleshooting

**`RuntimeError: DB_ROOT não está definido`**
O arquivo `.env` não foi criado ou `DB_ROOT` está vazio. Copie
`.env.example` para `.env` e preencha.

**`FileNotFoundError: ... .sp3.pkl`**
A etapa 3.3 (posições de satélite) precisa das efemérides SP3 em
`DB_ROOT/gnss/ephemeris/{year}/{doy}/`. O download automático de SP3 não
está implementado — baixe de um provedor como CDDIS e coloque os arquivos
lá antes de rodar.

**Downloads RAMSAC falhando com erro SSL**
É normal: o certificado do IGN costuma ter problema de cadeia. O código
já usa `verify=False` e silencia os warnings. Se falhar mesmo assim,
verifique se o seu firewall permite conexão HTTPS para
`ramsac.ign.gob.ar`.

**Muitos arquivos acabando em `tec_error/`**
Abra um deles com `pickle.load` e inspecione o header. Geralmente o RINEX
está corrompido ou usa um identificador de observação não reconhecido.
Arquivos RINEX 2 com convenções antigas (C1 isolado, sem P1) podem não
gerar resultados.

**Parquet fica com colunas antigas depois que mudei a lista de estações**
O `save_prn_incremental` é aditivo por design — ele nunca remove colunas
existentes. Para limpar, apague manualmente os arquivos em
`DB_ROOT/ionosphere/tec_pc/{year}/{doy}/` antes de reprocessar.

---

## Referências

- **Formato RINEX v3**: IGS, *RINEX: The Receiver Independent Exchange
  Format*, v3.04. https://files.igs.org/pub/data/format/rinex304.pdf
- **Formato SP3-d**: Hilla, S., *The Extended Standard Product 3 Orbit
  Format (SP3-d)*, NGS, 2016.
- **Compressão Hatanaka (CRINEX)**: Hatanaka, Y. (2008), *A Compression
  Format and Tools for GNSS Observation Data*, Bull. Geogr. Surv. Inst.,
  55.
- **TEC a partir de GNSS**: Hofmann-Wellenhof, B., Lichtenegger, H. &
  Wasle, E. (2008), *GNSS — Global Navigation Satellite Systems*,
  Springer, cap. 6.
- **Modelo de camada fina**: Schaer, S. (1999), *Mapping and predicting
  the Earth's ionosphere using the Global Positioning System*, tese,
  Univ. Bern.
- **RBMC**: https://www.ibge.gov.br/geociencias/informacoes-sobre-posicionamento-geodesico/rede-geodesica/16327-rede-brasileira-de-monitoramento-continuo-dos-sistemas-gnss-rbmc.html
- **RAMSAC**: https://www.ign.gob.ar/NuestrasActividades/Geodesia/Ramsac

---

## Citação

Se este pipeline for útil no seu trabalho, por favor cite:

```bibtex
@software{gnss_tec_pipeline,
  author = {Honda, R. H.},
  title  = {gnss-tec-pipeline: A GNSS-to-TEC processing pipeline for RBMC and RAMSAC},
  year   = {2025},
  url    = {https://github.com/<seu-usuario>/gnss-tec-pipeline}
}
```

---

## Autores e créditos

- **Rodrigo Hisashi Honda** — autor original dos módulos `rinex`, `sp3` e
  `tec` (2022–2023) e do pipeline completo.
- Reorganização do repositório e limpeza de código: 2026.

---

## Licença

TBD. Enquanto a licença não for definida, o repositório é considerado
privado e proprietário; entre em contato com o autor antes de qualquer
redistribuição.
