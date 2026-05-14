# Adaptive Weight Optimization of a Drought Precursor Index for Early Warning Applications

This repository provides the Python code used for the study:

**Adaptive Weight Optimization of a Drought Precursor Index for Early Warning Applications**

The code implements a weight optimization framework for the Drought Precursor Index (DPI), which integrates three meteorological indicators: precipitation deficit, vapor pressure deficit (VPD), and wind-speed anomaly. The purpose of the code is to identify optimal DPI weight combinations for drought early warning by considering detection performance, warning lead time, and signal persistence.

## Overview

The weighted DPI is calculated as:

```text
DPI_w = wP × (-Z_P) + wV × Z_VPD + wW × Z_WS
```

where:

- `wP` = precipitation-deficit weight
- `wV` = VPD weight
- `wW` = wind-speed weight
- `Z_P`, `Z_VPD`, and `Z_WS` are standardized meteorological components

The weights are constrained as:

```text
wP + wV + wW = 1
```

The script evaluates feasible weight combinations and performs:

- response surface analysis
- Pareto-based multi-objective optimization
- sensitivity analysis

## Input Data

The code requires a drought event file:

```text
drought_events.csv
```

with the following columns:

```text
id
event_time
```

The code also requires daily meteorological data files for each drought event:

```text
kma_daily_event_1.csv
kma_daily_event_2.csv
...
kma_daily_event_10.csv
```

Each meteorological file should include:

```text
date_kst
rn_day_2355
ta_0900
hm_0900
ws_10m_0900
```

These variables correspond to daily precipitation, air temperature, relative humidity, and wind speed.

## Main Settings

The main analysis settings can be modified in the `USER SETTINGS` section of the Python script:

```python
P_WINDOW = 90
V_WINDOW = 7
W_WINDOW = 7

COMPONENT_LOOKBACK_DAYS = 180
METRIC_LOOKBACK_DAYS = 90

WEIGHT_STEP = 0.05
DPI_THRESHOLD = 0.9
CONSEC_DAYS = 3
```

Before running the code, update the input and output paths:

```python
EVENTS_CSV = Path("path/to/drought_events.csv")
KMA_DIR = Path("path/to/kma_daily_outputs")
OUT_DIR = Path("path/to/output_directory")
```

## Requirements

The code requires Python 3 and the following packages:

```text
numpy
pandas
matplotlib
```

Install the required packages using:

```bash
pip install numpy pandas matplotlib
```

## How to Run

After preparing the input files and modifying the file paths, run:

```bash
python weightfactor.py
```

The output tables and figures will be saved in the specified output directory.

## Outputs

The script generates summary tables and figures, including:

```text
weight_grid.csv
metrics_by_event_and_weight.csv
weight_optimization_summary.csv
top20_weight_combinations.csv
pareto_flagged_weight_summary.csv
pareto_front_lead_auc_duration.png
ternary_response_surface_composite.png
response_surface_contour_wP_wV.png
response_map_lead_time.png
response_map_auc.png
response_map_duration.png
response_map_composite.png
sensitivity_wP_composite.png
sensitivity_wV_composite.png
sensitivity_wW_composite.png
sensitivity_importance_tornado.png
```
