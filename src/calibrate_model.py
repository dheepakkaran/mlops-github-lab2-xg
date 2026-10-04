import argparse

import joblib
from sklearn.calibration import CalibratedClassifierCV

from data_utils import FEATURES, MODELS_DIR, TARGET, load_splits, model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None)
    parser.add_argument("--method", default="sigmoid", choices=["sigmoid", "isotonic"])
    args = parser.parse_args()

    version = args.version or (MODELS_DIR / "latest_version.txt").read_text().strip()
    model = joblib.load(model_path(version))
    _, calib, _ = load_splits()

    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=args.method)
    except ImportError:
        calibrated = CalibratedClassifierCV(model, method=args.method, cv="prefit")

    calibrated.fit(calib[FEATURES], calib[TARGET])
    joblib.dump(calibrated, model_path(version, calibrated=True))
    print(f"Calibrated {version} with {args.method} on {len(calib)} shots")


if __name__ == "__main__":
    main()
