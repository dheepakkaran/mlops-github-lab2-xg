import argparse
import json

import joblib
from sklearn.metrics import brier_score_loss, f1_score, log_loss, roc_auc_score

from data_utils import FEATURES, METRICS_DIR, TARGET, load_splits, model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--calibrated", action="store_true")
    args = parser.parse_args()

    _, _, test = load_splits()
    model = joblib.load(model_path(args.version, args.calibrated))
    y = test[TARGET]
    p = model.predict_proba(test[FEATURES])[:, 1]

    sb = test.dropna(subset=["statsbomb_xg"])
    metrics = {
        "version": args.version,
        "calibrated": args.calibrated,
        "n_test_shots": len(test),
        "f1": round(f1_score(y, (p >= 0.5).astype(int)), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "log_loss": round(log_loss(y, p), 4),
        "mean_predicted_xg": round(float(p.mean()), 4),
        "actual_goal_rate": round(float(y.mean()), 4),
        "statsbomb_xg_brier": round(brier_score_loss(sb[TARGET], sb["statsbomb_xg"]), 4),
    }

    METRICS_DIR.mkdir(exist_ok=True)
    suffix = "_calibrated" if args.calibrated else ""
    out = METRICS_DIR / f"{args.version}{suffix}_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
