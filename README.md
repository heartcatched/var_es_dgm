## This repository contains the code for the thesis "The estimation of Value-at-Risk and Expected Shortfall based on deep generative models"
Author:  **Belonovskiy Peter Ilich, HSE DSBA** 

Supervisor: **Naumenko Vladimir Vladimirovich, HSE Associate Professor**

Thesis: https://www.hse.ru/en/edu/vkr/926006463?ysclid=m3e6gvwzru724833110

### Installation
___
The dependency managament in project was implemented via [poetry](https://python-poetry.org).
To clone this repository and set up the environment, run the following commands:
```bash
git clone https://github.com/BELONOVSKII/var_es_dgm.git
cd var_es_dgm
poetry install
poetry shell
```
**Note:** poetry should be pre-installed in your system.

### Download data
___
Thesis uses daily stock prices data from yahoo finance. To parse the yahoo finance and download data run:
```python
python var_es_dgm/data_parcing/parse_yfinance.py 
```
This downloads individual stocks's data and produces combined file `data/complete_stocks.csv` that would be further used in the experiments.

### Models
___
* **Variance-Covariance** (parametric Gaussian): `var_es_dgm/basic_models/parametric.py`
* **Historical Simulation**: `var_es_dgm/basic_models/hist_sim.py`
* **GARCH(1,1)**: `var_es_dgm/experiments/cli.py` (`evaluate_garch`)
* **DeepAR** (Normal and Student-t output distributions): `var_es_dgm/basic_models/deepar.py`
* **DeepVaR** (Fatouros et al., wraps DeepAR Student-t): `var_es_dgm/basic_models/deepvar.py`
* **TimeGrad** (LSTM + DDPM diffusion): `var_es_dgm/TimeGrad/`
* **TimeGrad tuned** — same architecture with Optuna-optimised hyperparameters (hardcoded in `cli.py:TUNED_TIMEGRAD`)

### Running Experiments
___
Experiments are run via the CLI in `var_es_dgm/experiments/cli.py`.

Run a single method:
```bash
python -m var_es_dgm.experiments.cli --dimension univariate --method timegrad --level 0.05
```

Run all methods for one dimension/level:
```bash
python -m var_es_dgm.experiments.cli --dimension univariate --method all --level 0.05
```

Run the full 2×8×2 grid (all dimensions, methods, and alpha levels):
```bash
python -m var_es_dgm.experiments.cli --run-all
```

Available `--method` values: `timegrad`, `timegrad_tuned`, `historical`, `variance_covariance`, `deepar_normal`, `deepar_student`, `deepvar`, `garch`.  
Available `--dimension` values: `univariate`, `multivariate`.  
Available `--level` values: `0.01`, `0.05`.

Key optional flags: `--device cuda`, `--n-repeats 5`, `--portfolio-size 10`, `--results-dir results/`, `--train-cutoff 2022-06-01`.

Results are saved under `{results_dir}/logs|checkpoints|results/{dimension}_{method}_{level}/`.

Jupyter notebooks with manual experiment runs are also available in `experiments/univariate` and `experiments/multivariate`.

### Hyperparameter Tuning
___
TimeGrad hyperparameters are tuned with [Optuna](https://optuna.org) via `var_es_dgm/experiments/tune.py`:
```bash
python -m var_es_dgm.experiments.tune --dimension univariate --level 0.05
```
The best configs found are hardcoded in `cli.py:TUNED_TIMEGRAD` and used automatically when `--method timegrad_tuned`.

### Visualisations
___
All figures from the thesis could be created by running notebooks in `visualisations/`.
