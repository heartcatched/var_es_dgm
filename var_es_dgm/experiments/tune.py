"""Optuna hyperparameter tuning for TimeGrad VaR/ES estimation."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math

import numpy as np
import optuna
import pandas as pd
import torch
from diffusers import DDPMScheduler

from ..TimeGrad import TimeGrad
from ..stat_tests import generate_report
from ..utils import (
    estimate_var_es_torch,
    estimate_var_es_torch_multivariate,
    seed_everything,
)
from .cli import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT,
    DEFAULT_CUTOFF,
    DEFAULT_DATA_PATH,
    DEFAULT_N_SAMPLES,
    DEFAULT_PORTFOLIO,
    DEFAULT_SEED,
    DEFAULT_TRAIN_SAMPLES,
    DIMENSIONS,
    LEVELS,
    PortfolioData,
    build_timegrad_loader,
    load_market_data,
    prepare_portfolio,
    select_tickers,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_VAL_DAYS = 126  
DEFAULT_N_TRIALS = 50



# Validation evaluation
def evaluate_on_val(
    model: TimeGrad,
    portfolio: PortfolioData,
    dimension: str,
    alpha: float,
    val_size: int,
    device: str,
) -> Dict[str, float]:
    """Run VaR/ES evaluation on the first val_size test observations."""
    VaR = torch.zeros(val_size)
    ES = torch.zeros(val_size)
    model.eval()
    for idx in range(val_size):
        context = portfolio.test_context[idx : idx + 1]
        if dimension == "univariate":
            var_i, es_i = estimate_var_es_torch(
                model, context, alpha=alpha, n_samples=DEFAULT_N_SAMPLES, device=device
            )
        else:
            corr = torch.corrcoef(torch.squeeze(context).T).to(torch.double)
            var_i, es_i = estimate_var_es_torch_multivariate(
                model,
                context,
                scaler=portfolio.scaler,
                R=corr,
                alpha=alpha,
                n_samples=DEFAULT_N_SAMPLES,
                device=device,
            )
        VaR[idx], ES[idx] = var_i, es_i

    metric_target = (
        portfolio.test_target_scaled[:val_size]
        if dimension == "univariate"
        else portfolio.test_target_real[:val_size]
    )
    return {k: float(v) for k, v in generate_report(metric_target, VaR, ES, alpha).items()}


# Composite objective score (higher = better)
def composite_score(metrics: Dict[str, float]) -> float:
    """Kupiec POF p-value as primary objective, with symmetric ES calibration penalty.

    - Kupiec p-value: higher means VaR violation rate matches alpha (don't reject H0).
    - Acerbi Z2 ≈ 0 under H0; Z2 < 0 means ES is underestimated (risky),
      Z2 > 0 means ES is overestimated (model too conservative, too few violations).
      We penalise both directions: underestimation fully, overestimation at 0.5 weight.
    """
    kupiec = metrics.get("Kupicks POF", 0.0)
    z2 = metrics.get("Acerbi Szekely 2", 0.0)
    if math.isnan(z2):
        z2 = 0.0
    penalty = min(0.0, z2) - max(0.0, z2) * 0.5
    return kupiec + penalty



# Per-portfolio train + score helper
def _train_and_score(
    params: Dict,
    portfolio: PortfolioData,
    dimension: str,
    alpha: float,
    val_size: int,
    device: str,
    trial_seed: int,
) -> Tuple[float, Dict[str, float]]:
    """Train one model and return (composite_score, metrics). Raises on failure."""
    num_train_samples = min(
        DEFAULT_TRAIN_SAMPLES, portfolio.train_tensor.shape[0] - DEFAULT_CONTEXT
    )
    loader = build_timegrad_loader(
        portfolio.train_tensor,
        context_size=DEFAULT_CONTEXT,
        num_samples=num_train_samples,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=trial_seed,
    )
    scheduler = DDPMScheduler(
        num_train_timesteps=params["num_train_timesteps"],
        beta_end=params["beta_end"],
        clip_sample=False,
    )
    model = TimeGrad(
        target_dim=portfolio.feature_dim if dimension == "multivariate" else 1,
        input_size=portfolio.feature_dim if dimension == "multivariate" else 1,
        scheduler=scheduler,
        hidden_size=50,
        num_layers=2,
        dropout_rate=params["dropout_rate"],
        num_inference_steps=params["num_train_timesteps"],
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    model.fit(loader, optimizer, params["n_epochs"], device, verbose=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = evaluate_on_val(model, portfolio, dimension, alpha, val_size, device)

    return composite_score(metrics), metrics



# Optuna objective factory
def make_objective(
    portfolios: List[Tuple[PortfolioData, int]],
    dimension: str,
    alpha: float,
    device: str,
    base_seed: int,
):
    """Return a closure that Optuna calls for each trial.

    Parameters
    ----------
    portfolios:
        List of (PortfolioData, val_size) tuples. The trial score is the mean
        composite score across all portfolios.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_train_timesteps": trial.suggest_int("num_train_timesteps", 10, 100),
            "beta_end": trial.suggest_float("beta_end", 0.05, 0.5, log=True),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "n_epochs": trial.suggest_int("n_epochs", 20, 100),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.6),
        }

        scores = []
        all_metrics: List[Dict[str, float]] = []
        for p_idx, (portfolio, val_size) in enumerate(portfolios):
            trial_seed = base_seed + trial.number * 100 + p_idx
            seed_everything(trial_seed)
            try:
                score, metrics = _train_and_score(
                    params, portfolio, dimension, alpha, val_size, device, trial_seed
                )
                scores.append(score)
                all_metrics.append(metrics)
            except Exception as exc:
                trial.set_user_attr(f"error_p{p_idx}", str(exc))
                scores.append(float("-inf"))

        if all(s == float("-inf") for s in scores):
            return float("-inf")

        mean_score = float(np.mean([s for s in scores if s != float("-inf")]))

        valid = [m for m, s in zip(all_metrics, scores) if s != float("-inf")]
        if valid:
            for k in valid[0]:
                trial.set_user_attr(k, float(np.mean([m[k] for m in valid])))

        return mean_score

    return objective


# Main tuning entry point
def run_tuning(
    dimension: str,
    alpha: float,
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    results_dir: Path = Path("results"),
    device: str = "cpu",
    n_trials: int = DEFAULT_N_TRIALS,
    n_portfolios: int = 1,
    portfolio_size: int = DEFAULT_PORTFOLIO,
    seed: int = DEFAULT_SEED,
    train_cutoff: str = DEFAULT_CUTOFF,
    val_days: int = DEFAULT_VAL_DAYS,
    study_name: Optional[str] = None,
) -> Dict:
    if dimension not in DIMENSIONS:
        raise ValueError(f"Unsupported dimension: {dimension}")
    if alpha not in LEVELS:
        raise ValueError(f"Unsupported alpha: {alpha}")

    df = load_market_data(data_path)
    cutoff = pd.Timestamp(train_cutoff)
    tune_cutoff = cutoff - pd.tseries.offsets.BDay(val_days)

    portfolios: List[Tuple[PortfolioData, int]] = []
    all_tickers: List[List[str]] = []
    for i in range(n_portfolios):
        p_seed = seed + i
        tickers = select_tickers(df, portfolio_size, p_seed)
        seed_everything(p_seed)
        portfolio = prepare_portfolio(df, tickers, dimension, DEFAULT_CONTEXT, tune_cutoff)
        actual_val_size = min(val_days, portfolio.test_size)
        if actual_val_size < 20:
            raise ValueError(
                f"Validation window too small ({actual_val_size} obs) for portfolio {i}. "
                "Try a smaller val_days or an earlier train_cutoff."
            )
        portfolios.append((portfolio, actual_val_size))
        all_tickers.append(tickers)

    print(
        f"Tuning TimeGrad [{dimension}, alpha={alpha:.0%}] | "
        f"portfolios: {n_portfolios} | "
        f"train obs (p0): {portfolios[0][0].train_tensor.shape[0]} | "
        f"val obs: {portfolios[0][1]} | "
        f"trials: {n_trials}"
    )

    name = study_name or f"timegrad_{dimension}_{int(alpha * 100)}pct"
    study = optuna.create_study(direction="maximize", study_name=name)
    study.optimize(
        make_objective(portfolios, dimension, alpha, device, seed),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    result = {
        "dimension": dimension,
        "alpha": alpha,
        "best_score": best.value,
        "best_params": best.params,
        "best_metrics": {k: v for k, v in best.user_attrs.items() if "error" not in k},
        "tickers": all_tickers,
        "tune_cutoff": str(tune_cutoff.date()),
        "train_cutoff": train_cutoff,
        "val_days": portfolios[0][1],
        "n_portfolios": n_portfolios,
        "n_trials": n_trials,
    }

    out_dir = results_dir / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"\nBest score : {best.value:.4f}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"Saved to   : {out_path}")

    return result

# CLI

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Tune TimeGrad with Optuna.")
    parser.add_argument("--dimension", choices=DIMENSIONS, required=True)
    parser.add_argument("--level", type=float, choices=LEVELS, default=0.05)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument(
        "--n-portfolios",
        type=int,
        default=1,
        help="Number of portfolios to average the score over per trial (default: 1).",
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=DEFAULT_VAL_DAYS,
        help="Number of trading days to use as validation (default: 126 ≈ 6 months).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--portfolio-size", type=int, default=DEFAULT_PORTFOLIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--study-name", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_tuning(
        dimension=args.dimension,
        alpha=args.level,
        data_path=args.data_path,
        results_dir=args.results_dir,
        device=args.device,
        n_trials=args.n_trials,
        n_portfolios=args.n_portfolios,
        portfolio_size=args.portfolio_size,
        seed=args.seed,
        train_cutoff=args.train_cutoff,
        val_days=args.val_days,
        study_name=args.study_name,
    )


if __name__ == "__main__":
    main()
