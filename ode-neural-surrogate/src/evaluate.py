import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score

from .config import INPUT_COLS, OUTPUT_COLS


def evaluate_model(model, X_test_t, Y_test_t, y_scaler, output_file=None):
    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(X_test_t).numpy()

    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(Y_test_t.numpy())

    rows = []
    print("\nTest-set performance:\n")

    for i, label in enumerate(OUTPUT_COLS):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))

        rows.append({"output": label, "r2": r2, "rmse": rmse})

        print(label)
        print(f"  R2   = {r2:.5f}")
        print(f"  RMSE = {rmse:.6f}\n")

    metrics_df = pd.DataFrame(rows)
    if output_file:
        metrics_df.to_csv(output_file, index=False)

    return metrics_df


def predict_all(model, dataset, x_scaler, y_scaler, output_file=None):
    model.eval()

    with torch.no_grad():
        x_all = dataset[INPUT_COLS].values
        x_all_scaled = x_scaler.transform(x_all)
        x_all_t = torch.tensor(x_all_scaled, dtype=torch.float32)

        y_all_scaled = model(x_all_t).numpy()
        y_all = y_scaler.inverse_transform(y_all_scaled)

    data_cmp = dataset.copy()

    states = ["H", "hH", "oH", "aH", "dH", "Ala"]

    for i, state in enumerate(states):
        data_cmp[f"{state}_nn_mM"] = y_all[:, i]
        data_cmp[f"{state}_error_mM"] = data_cmp[f"{state}_nn_mM"] - data_cmp[f"{state}_mM"]

    if output_file:
        data_cmp.to_csv(output_file, index=False)

    return data_cmp


def save_final_comparison(data_cmp, output_file=None):
    rows = []

    for (f1, keq2), df in data_cmp.groupby(["f1_mM_min", "Keq2_dimensionless"]):
        df = df.sort_values("time_min")

        row = {
            "f1_mM_min": f1,
            "Keq2_dimensionless": keq2,
        }

        for state in ["H", "hH", "oH", "aH", "dH", "Ala"]:
            row[f"{state}_ode_mM"] = df[f"{state}_mM"].iloc[-1]
            row[f"{state}_nn_mM"] = df[f"{state}_nn_mM"].iloc[-1]
            row[f"{state}_error_mM"] = df[f"{state}_error_mM"].iloc[-1]

        row["total_analytes_mM"] = df["total_analytes_mM"].iloc[-1]
        row["expected_HAME_feed_mM"] = df["expected_HAME_feed_mM"].iloc[-1]
        row["mass_balance_error_mM"] = df["mass_balance_error_mM"].iloc[-1]
        row["expected_Ala_mM"] = df["expected_Ala_mM"].iloc[-1]
        row["Ala_balance_error_mM"] = df["Ala_balance_error_mM"].iloc[-1]

        rows.append(row)

    final_cmp_df = pd.DataFrame(rows)

    if output_file:
        final_cmp_df.to_csv(output_file, index=False)

    return final_cmp_df
