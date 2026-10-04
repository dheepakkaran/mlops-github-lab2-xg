# xG model report - `20261004235641-7cdb318`

Test set: 547 non-penalty shots (World Cup 2022 + Euro 2024).

| Metric | Uncalibrated | Calibrated |
|---|---|---|
| roc_auc | 0.8006 | 0.8006 |
| brier | 0.1253 | 0.0726 |
| log_loss | 0.4016 | 0.2547 |
| mean_predicted_xg | 0.2939 | 0.0827 |
| actual_goal_rate | 0.0914 | 0.0914 |
| f1 | 0.4125 | 0.0 |
| tuned_threshold | 0.63 | 0.18 |
| f1_tuned | 0.3604 | 0.3636 |

StatsBomb xG Brier score on the same shots (benchmark): **0.0646**

![Calibration curve](calibration_curve.png)

![xG shot map](xg_shot_map.png)