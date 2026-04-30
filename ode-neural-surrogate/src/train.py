import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import INPUT_COLS, OUTPUT_COLS
from .dataset import add_toxicity_flag, final_time_dataframe, generate_dataset
from .evaluate import evaluate_model, predict_all, save_final_comparison
from .nn_model import TinyMLP
from .plots import plot_all_final_heatmaps, plot_balance_checks, plot_example_trajectory, plot_heatmap


def prepare_tensors(dataset, test_size=0.2, random_state=42):
    sim_ids = dataset["sim_id"].unique()
    train_ids, test_ids = train_test_split(
        sim_ids,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    train_df = dataset[dataset["sim_id"].isin(train_ids)].reset_index(drop=True)
    test_df = dataset[dataset["sim_id"].isin(test_ids)].reset_index(drop=True)

    X_train = train_df[INPUT_COLS].values
    Y_train = train_df[OUTPUT_COLS].values

    X_test = test_df[INPUT_COLS].values
    Y_test = test_df[OUTPUT_COLS].values

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    Y_train_scaled = y_scaler.fit_transform(Y_train)
    Y_test_scaled = y_scaler.transform(Y_test)

    tensors = {
        "X_train": torch.tensor(X_train_scaled, dtype=torch.float32),
        "Y_train": torch.tensor(Y_train_scaled, dtype=torch.float32),
        "X_test": torch.tensor(X_test_scaled, dtype=torch.float32),
        "Y_test": torch.tensor(Y_test_scaled, dtype=torch.float32),
    }

    return train_df, test_df, train_ids, test_ids, x_scaler, y_scaler, tensors


def train_model(
    X_train_t,
    Y_train_t,
    input_dim,
    epochs=80,
    batch_size=2048,
    lr=1e-3,
    seed=42,
):
    torch.manual_seed(seed)

    model = TinyMLP(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train_t.size(0))
        epoch_losses = []

        for i in range(0, X_train_t.size(0), batch_size):
            idx = perm[i : i + batch_size]

            optimizer.zero_grad()
            pred = model(X_train_t[idx])
            loss = loss_fn(pred, Y_train_t[idx])
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        history.append({"epoch": epoch, "loss": mean_loss})

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss = {mean_loss:.6f}")

    return model, pd.DataFrame(history)


def run_pipeline(
    output_dir,
    epochs=80,
    batch_size=2048,
    lr=1e-3,
    skip_plots=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset(output_file=output_dir / "kinetic_dataset_Keq2_f1.csv")

    final_df = final_time_dataframe(dataset, output_file=output_dir / "final_ODE_results.csv")
    final_df = add_toxicity_flag(final_df)
    final_df.to_csv(output_dir / "final_ODE_results_with_toxicity.csv", index=False)

    print("\nToxicity check:")
    print(final_df["H_above_toxic_limit"].value_counts())

    train_df, test_df, train_ids, test_ids, x_scaler, y_scaler, tensors = prepare_tensors(dataset)

    model, history = train_model(
        tensors["X_train"],
        tensors["Y_train"],
        input_dim=tensors["X_train"].shape[1],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    history.to_csv(output_dir / "training_history.csv", index=False)

    torch.save(model.state_dict(), model_dir / "tiny_mlp_surrogate.pt")
    joblib.dump(x_scaler, model_dir / "x_scaler.joblib")
    joblib.dump(y_scaler, model_dir / "y_scaler.joblib")

    metrics_df = evaluate_model(
        model,
        tensors["X_test"],
        tensors["Y_test"],
        y_scaler,
        output_file=output_dir / "test_metrics.csv",
    )

    data_cmp = predict_all(
        model,
        dataset,
        x_scaler,
        y_scaler,
        output_file=output_dir / "full_ODE_NN_error_all_timepoints.csv",
    )

    final_cmp_df = save_final_comparison(
        data_cmp,
        output_file=output_dir / "final_comparison_ODE_NN_errors.csv",
    )

    if not skip_plots:
        plot_balance_checks(final_df, output_dir=output_dir)
        plot_heatmap(
            final_df,
            "total_analytes_mM",
            "Final total analytes",
            "mM",
            output_dir / "heatmap_total_analytes.png",
        )
        plot_heatmap(
            final_df,
            "mass_balance_error_mM",
            "Final HAME mass balance error",
            "mM",
            output_dir / "heatmap_mass_balance_error.png",
        )
        plot_example_trajectory(
            model,
            test_df,
            test_ids[0],
            x_scaler,
            y_scaler,
            output_dir / "example_trajectory_ODE_vs_NN.png",
        )
        plot_all_final_heatmaps(final_cmp_df, output_dir=output_dir)

    print("\nDone. Outputs saved to:", output_dir)
    return metrics_df


def parse_args():
    parser = argparse.ArgumentParser(description="Train ODE neural surrogate model.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        skip_plots=args.skip_plots,
    )
