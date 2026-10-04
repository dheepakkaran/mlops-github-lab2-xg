import argparse
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from data_utils import FEATURES, METRICS_DIR, REPORTS_DIR, TARGET, load_splits, model_path


def calibration_plot(test, raw, cal, version):
    y = test[TARGET]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    curves = [
        ("Uncalibrated (balanced GB)", raw.predict_proba(test[FEATURES])[:, 1], "o-"),
        ("Calibrated (ours)", cal.predict_proba(test[FEATURES])[:, 1], "o-"),
        ("StatsBomb xG", test["statsbomb_xg"].fillna(0).to_numpy(), "s:"),
    ]
    for label, p, style in curves:
        frac_goals, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac_goals, style, label=label)
    ax.set(xlabel="Predicted xG", ylabel="Actual goal rate",
           title=f"Calibration curve - {version}", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "calibration_curve.png", dpi=120)
    plt.close(fig)


def shot_map(test, cal):
    p = cal.predict_proba(test[FEATURES])[:, 1]
    fig, ax = plt.subplots(figsize=(7, 6))
    pitch = dict(color="grey", lw=1)
    ax.plot([60, 120, 120, 60, 60], [0, 0, 80, 80, 0], **pitch)
    ax.plot([120, 102, 102, 120], [18, 18, 62, 62], **pitch)
    ax.plot([120, 114, 114, 120], [30, 30, 50, 50], **pitch)
    ax.plot([120, 120], [36, 44], color="black", lw=4)
    sc = ax.scatter(test["x"], test["y"], c=p, s=20 + 300 * p, cmap="viridis", alpha=0.75)
    goals = test[test[TARGET] == 1]
    ax.scatter(goals["x"], goals["y"], s=60, facecolors="none", edgecolors="red", label="Goal")
    fig.colorbar(sc, ax=ax, label="Calibrated xG")
    ax.set(xlim=(58, 122), ylim=(82, -2), aspect="equal", title="Test-set shots coloured by xG")
    ax.axis("off")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "xg_shot_map.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    v = args.version

    _, _, test = load_splits()
    raw = joblib.load(model_path(v))
    cal = joblib.load(model_path(v, calibrated=True))
    REPORTS_DIR.mkdir(exist_ok=True)
    calibration_plot(test, raw, cal, v)
    shot_map(test, cal)

    m_raw = json.loads((METRICS_DIR / f"{v}_metrics.json").read_text())
    m_cal = json.loads((METRICS_DIR / f"{v}_calibrated_metrics.json").read_text())
    rows = ["roc_auc", "brier", "log_loss", "mean_predicted_xg", "actual_goal_rate",
            "f1", "tuned_threshold", "f1_tuned"]
    md = [
        f"# xG model report - `{v}`", "",
        f"Test set: {m_cal['n_test_shots']} non-penalty shots (World Cup 2022 + Euro 2024).", "",
        "| Metric | Uncalibrated | Calibrated |", "|---|---|---|",
        *[f"| {k} | {m_raw[k]} | {m_cal[k]} |" for k in rows], "",
        f"StatsBomb xG Brier score on the same shots (benchmark): **{m_cal['statsbomb_xg_brier']}**", "",
        "![Calibration curve](calibration_curve.png)", "",
        "![xG shot map](xg_shot_map.png)",
    ]
    text = "\n".join(md)
    (REPORTS_DIR / "latest_report.md").write_text(text)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("\n".join(line for line in md if not line.startswith("![")))
    print(text)


if __name__ == "__main__":
    main()
