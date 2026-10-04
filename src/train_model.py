import argparse

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from data_utils import FEATURES, MODELS_DIR, TARGET, load_splits, model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    train, _, _ = load_splits()
    X, y = train[FEATURES], train[TARGET]
    weights = compute_sample_weight(class_weight="balanced", y=y)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
    )
    model.fit(X, y, sample_weight=weights)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, model_path(args.version))
    (MODELS_DIR / "latest_version.txt").write_text(args.version)
    print(f"Trained on {len(train)} shots ({int(y.sum())} goals) -> {model_path(args.version).name}")


if __name__ == "__main__":
    main()
