# GNSS Ionospheric Pipeline

Processing and visualization pipeline for dTEC (Total Electron Content variation) derived from GNSS observations. Computes Ionospheric Pierce Points (IPP) for each station/PRN pair and generates diagnostic figures.

---

## Project structure

```
project/
│
├── shared/                      # Shared utilities — reusable across all scripts
│   ├── __init__.py
│   ├── geometry.py              # IPP geometry (ipp_calc, ipp_geometric)
│   └── config.py                # Central path configuration (DB_ROOT, helpers)
│
├── rasther_dtec/
│   ├── rasther_dtec.py          # Main pipeline (process_prn_file, run_pipeline_and_plot)
│   └── figures_rasther.py       # Plotting functions specific to this routine
│
└── .env                         # DB_ROOT definition (not committed to version control)
```

---

## Modules

### `shared/config.py`

Central configuration for the entire pipeline. Defines `DB_ROOT` (the root of the database on disk) and exposes path-building helpers used by every script.

**Configuration:** Set the active `place` variable to match your machine:

```python
place = 'ssd-inpe'   # /media/hisashi/ESSD/database
place = 'inpe'       # /home/hisashi/database
place = 'home'       # D:\database
place = 'ssd-home'   # (to be defined)
```

**Path helpers:**

| Function | Returns |
|---|---|
| `crinex_dir(year, doy)` | `.../gnss/crinex/YYYY/DDD` |
| `rinex_dir(year, doy)` | `.../gnss/rinex/YYYY/DDD` |
| `ephemeris_dir(year, doy)` | `.../gnss/ephemeris/YYYY/DDD` |
| `station_positions_dir()` | `.../gnss/station_positions` |
| `tec_dir(year, doy)` | `.../ionosphere/tec/YYYY/DDD` |
| `tec_pc_dir(year, doy)` | `.../ionosphere/tec_pc/YYYY/DDD` |
| `dtec_pc_dir(year, doy)` | `.../ionosphere/dtec_pc/YYYY/DDD` |
| `tec_error_dir(year, doy)` | `.../ionosphere/tec_error/YYYY/DDD` |

**Utilities:**

- `doy_to_date(year, doy)` — converts day-of-year to `datetime`
- `gnss_week(date)` — returns the GNSS week number for a given date
- `setup_logging(level)` — configures pipeline-wide logging format

---

### `shared/geometry.py`

Pure geometry functions for IPP computation. No I/O, no plotting — only numpy. Safe to import in any script that needs ionospheric geometry.

#### `ipp_calc(prn_coordinates, station_coordinates, H_km, RE_km=6371.0)`

Computes the IPP using the classical **thin-shell trigonometric** approach (ENU rotation + zenith angle chi). Returns `(lat_pp_deg, lon_pp_deg, El_deg)`.

- Input satellite coordinates: ECEF in km `(x_p, y_p, z_p)`
- Input station coordinates: `(x_s, y_s, z_s, lon_deg, lat_deg, h)`
- Geographic clipping masks applied: lat > 20° and lon > −20° become NaN (South American sector focus — adjust as needed)

#### `ipp_geometric(prn_coordinates, station_coordinates, H_km, RE_km=6371.0)`

Alternative IPP formulation via **direct ray/sphere intersection** (solves the quadratic `‖r + λv‖² = (RE+H)²`). More robust numerically at large elevation angles. Same interface and return values as `ipp_calc`.

---

### `shared/__init__.py`

Currently empty. Can be populated with re-exports to shorten import paths:

```python
# example
from .geometry import ipp_calc, ipp_geometric
```

---

### `rasther_dtec/figures_rasther.py`

Plotting functions specific to the `rasther_dtec` routine. Not shared.

#### `plot_ipp_trajectory(data_pack, output_dir, cmap, trh)`

Generates a figure with two stacked panels (Time × Latitude on top, Time × Longitude on bottom) as a matplotlib scatter, **with convective storm overlays** (CPTEC c13 catalog). Each storm is drawn as a vertical line colored by minimum cloud top temperature (`Blues_r`, −95 to −65 °C).

- Output file: `{output_dir}/{date_string}_{prn}.png`
- Requires `df_storms` key in `data_pack`

#### `plot_lat_lon_scatter_mpl(data_pack, output_dir, cmap, trh, marker_size, marker, alpha, dpi)`

Simpler version — same two-panel layout but **no storm overlays**. Pure matplotlib scatter for quick dTEC signature inspection.

- Output file: `{output_dir}/{date_string}_{prn}_scatter_mpl.png`
- Returns the saved file path, or `None` if no valid points exist

**`data_pack` dictionary (expected keys for both functions):**

| Key | Content |
|---|---|
| `prn` | Satellite identifier string (e.g. `"G05"`) |
| `df_lt` | DataFrame — IPP latitudes (rows = time, cols = stations) |
| `df_ln` | DataFrame — IPP longitudes (rows = time, cols = stations) |
| `df_dtec` | DataFrame — dTEC values (same shape as df_lt/df_ln) |
| `time_index` | DatetimeIndex aligned with the DataFrames |
| `date_string` | String label used in titles and file names (e.g. `"2021-296"`) |
| `t_start` | Timestamp — start of the day window |
| `t_end` | Timestamp — end of the day window |
| `df_storms` | DataFrame with storm centroids *(only for plot_ipp_trajectory)* |

---

## Dependencies

```
numpy
pandas
matplotlib
datashader          # used by trajectories_to_pcolormesh (optional path)
fastparquet         # parquet engine used throughout
```

---

## Usage example

```python
from shared.config import dtec_pc_dir, station_positions_dir
from rasther_dtec.rasther_dtec import run_pipeline_and_plot

year, doy = 2021, 296

run_pipeline_and_plot(
    folder_base=dtec_pc_dir(year, doy),
    station_positions_file=station_positions_dir() / f"SP_{year}-{doy}.parquet",
    shell_height_km=200,
    time_freq="15s",
    cmap="turbo",
    trh=0.5,
    make_simple_scatter=True,   # produces figures_rasther: plot_lat_lon_scatter_mpl
    make_storm_figure=True,     # produces figures_rasther: plot_ipp_trajectory
)
```