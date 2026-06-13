"""Experimentos del frente de mejora del modelo, posteriores al tuning de Optuna.

Ejecuta, en orden y con trazabilidad completa en MLflow (tag ``technique``):

1. Comparación de esquemas de validación cruzada temporal sobre el mejor candidato.
2. Variantes de objetivo y de pesos de muestra para el sesgo en sílice alta.
3. Poda de features por permutation importance sobre validación.
4. Ensembles (blend ponderado y stacking temporal) con los modelos del stack.

Toda la selección ocurre sobre validación o walk-forward dentro de train+val. Este
script NO toca el test: la evaluación final del ganador se hace una sola vez aparte.
Los resultados quedan en reports/experiments/ para que el notebook 02 los presente.

Uso:  python scripts/run_experiments.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import load_config, resolve, set_seeds
from src.models.bias import bias_variants, silica_sample_weight, tercile_report
from src.models.cv import cross_validate_temporal, cv_summary
from src.models.delta import DeltaTargetRegressor
from src.models.ensemble import (
    blend_predict,
    fit_stacker,
    optimize_blend_weights,
    temporal_oof_predictions,
)
from src.models.evaluate import regression_metrics
from src.models.train import (
    git_commit,
    lag1_feature,
    make_lgbm,
    prepare_data,
    setup_mlflow,
)
from src.models.tuning import make_builder

import lightgbm as lgb

OUT = Path("reports/experiments")


def load_tuned() -> dict[str, dict]:
    """Lee los mejores parámetros de cada estudio de Optuna."""
    tuned = {}
    for path in resolve("reports/tuning").glob("best_*.json"):
        with open(path, encoding="utf-8") as fh:
            info = json.load(fh)
        tuned[f"{info['family']}_{info['representation']}"] = info
    return tuned


def candidate_builders(tuned: dict, seed: int, lag_col: str) -> dict[str, callable]:
    return {
        name: make_builder(info["family"], info["representation"], info["params"], seed, lag_col)
        for name, info in tuned.items()
    }


def log_run(name: str, technique: str, metrics: dict, params: dict | None = None,
            artifacts: list[str] | None = None) -> None:
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"technique": technique, "experiment_type": "improvement",
                         "git_commit": git_commit()})
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for art in artifacts or []:
            mlflow.log_artifact(art)


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    seed = cfg["seed"]
    set_seeds(seed)
    data = prepare_data(cfg)
    lag_col = lag1_feature(cfg)
    X_trval = pd.concat([data["X_train"], data["X_val"]])
    y_trval = pd.concat([data["y_train"], data["y_val"]])
    n_splits, gap = cfg["cv"]["n_splits"], cfg["cv"]["gap_hours"]

    out_dir = resolve(str(OUT))
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_mlflow(cfg)
    tuned = load_tuned()
    print("Candidatos tuneados:", {k: round(v["cv_mae_mean"], 4) for k, v in tuned.items()}, flush=True)
    best_name = min(tuned, key=lambda k: tuned[k]["cv_mae_mean"])
    best = tuned[best_name]
    print(f"Mejor candidato por CV: {best_name}", flush=True)

    # ------------------------------------------------------------------
    # 1. Esquemas de validación cruzada temporal (sobre el mejor candidato)
    # ------------------------------------------------------------------
    schemes = [("expanding", 0), ("sliding", 0), ("expanding", gap), ("sliding", gap)]
    scheme_rows = {}
    builder = make_builder(best["family"], best["representation"], best["params"], seed, lag_col)
    for scheme, g in schemes:
        label = f"{scheme}{'_gap' + str(g) if g else ''}"
        folds = cross_validate_temporal(builder, X_trval, y_trval, n_splits, scheme, g)
        scheme_rows[label] = {**cv_summary(folds), "folds": folds.reset_index().to_dict("records")}
        log_run(f"cv-{label}", "cv-scheme-comparison",
                {k: v for k, v in cv_summary(folds).items()},
                {"scheme": scheme, "gap": g, "candidate": best_name})
        print(f"  CV {label}: MAE {scheme_rows[label]['mae_mean']:.4f} "
              f"± {scheme_rows[label]['mae_std']:.4f}", flush=True)
    with open(out_dir / "cv_schemes.json", "w", encoding="utf-8") as fh:
        json.dump({"candidate": best_name, "schemes": scheme_rows}, fh, indent=2)

    # ------------------------------------------------------------------
    # 2. Sesgo en sílice alta: objetivos y pesos de muestra (en validación)
    # ------------------------------------------------------------------
    assert best["family"] == "lightgbm", "las variantes de sesgo están definidas para LightGBM"
    bias_rows = {}

    def fit_delta_lgbm(params: dict, weight_strength: float | None = None) -> DeltaTargetRegressor:
        base, stop = make_lgbm(params, seed)
        wrapper = DeltaTargetRegressor(base, lag_col)
        kw = {}
        if weight_strength is not None:
            kw["sample_weight"] = silica_sample_weight(data["y_train"], weight_strength)
        wrapper.fit(
            data["X_train"], data["y_train"],
            eval_set=[(data["X_val"], data["y_val"])],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stop, verbose=False)],
            **kw,
        )
        return wrapper

    variants: dict[str, tuple[dict, float | None]] = {
        name: (params, None) for name, params in bias_variants(best["params"]).items()
    }
    variants["peso_silice_x1"] = (best["params"], 1.0)
    variants["peso_silice_x2"] = (best["params"], 2.0)
    for name, (params, strength) in variants.items():
        model = fit_delta_lgbm(params, strength)
        preds = model.predict(data["X_val"])
        global_metrics = regression_metrics(data["y_val"], preds)
        terciles = tercile_report(data["y_val"], preds)
        bias_rows[name] = {
            "val_mae": global_metrics["mae"], "val_r2": global_metrics["r2"],
            "terciles": terciles.round(4).to_dict("index"),
        }
        log_run(f"bias-{name}", "bias-correction-high-silica",
                {"val_mae": global_metrics["mae"],
                 "val_mae_tercil_alto": float(terciles.loc["alto", "mae"]),
                 "val_sesgo_tercil_alto": float(terciles.loc["alto", "sesgo"])},
                {"variant": name})
        print(f"  sesgo {name}: val MAE {global_metrics['mae']:.4f} | "
              f"tercil alto MAE {terciles.loc['alto', 'mae']:.4f} "
              f"sesgo {terciles.loc['alto', 'sesgo']:+.4f}", flush=True)
    with open(out_dir / "bias_variants.json", "w", encoding="utf-8") as fh:
        json.dump(bias_rows, fh, indent=2)

    # ------------------------------------------------------------------
    # 3. Poda por permutation importance sobre validación
    # ------------------------------------------------------------------
    reference_model = fit_delta_lgbm(best["params"])
    ref_val_mae = regression_metrics(data["y_val"], reference_model.predict(data["X_val"]))["mae"]
    importance = permutation_importance(
        reference_model, data["X_val"], data["y_val"],
        scoring="neg_mean_absolute_error", n_repeats=5, random_state=seed,
    )
    imp = pd.Series(importance.importances_mean, index=data["X_val"].columns)
    selected = sorted(imp[imp > 0].index.tolist())
    if lag_col not in selected:
        selected.append(lag_col)  # el ancla del delta no puede salir
    base, stop = make_lgbm(best["params"], seed)
    pruned_model = DeltaTargetRegressor(base, lag_col)
    pruned_model.fit(
        data["X_train"][selected], data["y_train"],
        eval_set=[(data["X_val"][selected], data["y_val"])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stop, verbose=False)],
    )
    pruned_val_mae = regression_metrics(
        data["y_val"], pruned_model.predict(data["X_val"][selected])
    )["mae"]
    print(f"  poda: {len(imp)} -> {len(selected)} features | val MAE "
          f"{ref_val_mae:.4f} -> {pruned_val_mae:.4f}", flush=True)
    imp.sort_values(ascending=False).to_csv(out_dir / "permutation_importance.csv")
    with open(out_dir / "pruning.json", "w", encoding="utf-8") as fh:
        json.dump({"n_total": int(len(imp)), "n_selected": len(selected),
                   "val_mae_full": ref_val_mae, "val_mae_pruned": pruned_val_mae,
                   "selected": selected}, fh, indent=2)
    log_run("pruning-permutation", "permutation-pruning",
            {"val_mae_full": ref_val_mae, "val_mae_pruned": pruned_val_mae,
             "n_selected": len(selected)},
            artifacts=[str(out_dir / "permutation_importance.csv")])

    # ------------------------------------------------------------------
    # 4. Ensembles: blend ponderado y stacking temporal
    # ------------------------------------------------------------------
    builders = candidate_builders(tuned, seed, lag_col)
    fitted = {name: b(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
              for name, b in builders.items()}
    preds_val = {name: m.predict(data["X_val"]) for name, m in fitted.items()}
    singles = {name: regression_metrics(data["y_val"], p)["mae"] for name, p in preds_val.items()}
    print("  individuales (val MAE):", {k: round(v, 4) for k, v in singles.items()}, flush=True)

    weights = optimize_blend_weights(preds_val, data["y_val"])
    blend_mae = regression_metrics(data["y_val"], blend_predict(preds_val, weights))["mae"]
    print(f"  blend {weights}: val MAE {blend_mae:.4f}", flush=True)

    oof, y_oof = temporal_oof_predictions(
        builders, data["X_train"], data["y_train"], n_splits, gap)
    stacker = fit_stacker(oof, y_oof)
    stack_val = stacker.predict(pd.DataFrame(preds_val)[oof.columns])
    stack_mae = regression_metrics(data["y_val"], stack_val)["mae"]
    print(f"  stacking Ridge (coefs {dict(zip(oof.columns, stacker.coef_.round(3)))}): "
          f"val MAE {stack_mae:.4f}", flush=True)

    best_single_mae = min(singles.values())
    ensemble_summary = {
        "singles_val_mae": singles,
        "blend": {"weights": weights, "val_mae": blend_mae},
        "stacking": {"coefs": dict(zip(oof.columns, [float(c) for c in stacker.coef_])),
                     "val_mae": stack_mae},
        "best_single_val_mae": best_single_mae,
        "blend_rel_improvement_pct": 100 * (best_single_mae - blend_mae) / best_single_mae,
        "stack_rel_improvement_pct": 100 * (best_single_mae - stack_mae) / best_single_mae,
    }
    with open(out_dir / "ensembles.json", "w", encoding="utf-8") as fh:
        json.dump(ensemble_summary, fh, indent=2)
    log_run("ensemble-blend", "ensemble-blend",
            {"val_mae": blend_mae, "best_single_val_mae": best_single_mae})
    log_run("ensemble-stacking", "ensemble-stacking",
            {"val_mae": stack_mae, "best_single_val_mae": best_single_mae})

    print(f"Listo en {(time.time() - t0) / 60:.1f} min. Resultados en {out_dir}", flush=True)


if __name__ == "__main__":
    main()
