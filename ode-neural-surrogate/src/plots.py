from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import INPUT_COLS


def plot_balance_checks(final_df, output_dir):
    output_dir = Path(output_dir)

    plt.figure(figsize=(8, 5))
    plt.scatter(final_df["expected_HAME_feed_mM"], final_df["total_analytes_mM"])
    plt.xlabel("Expected HAME feed [mM]")
    plt.ylabel("Total analytes [mM]")
    plt.title("HAME mass balance check")
    plt.tight_layout()
    plt.savefig(output_dir / "hame_mass_balance_check.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(final_df["expected_Ala_mM"], final_df["Ala_mM"])
    plt.xlabel("Expected alanine [mM]")
    plt.ylabel("ODE alanine [mM]")
    plt.title("Alanine balance check")
    plt.tight_layout()
    plt.savefig(output_dir / "alanine_balance_check.png", dpi=300)
    plt.close()


def plot_heatmap(final_df, value_col, title, cbar_label, filename):
    grid = final_df.pivot(
        index="f1_mM_min",
        columns="Keq2_dimensionless",
        values=value_col,
    )

    plt.figure(figsize=(8, 5))
    im = plt.imshow(
        grid.values,
        origin="lower",
        aspect="auto",
        extent=[
            grid.columns.min(),
            grid.columns.max(),
            grid.index.min(),
            grid.index.max(),
        ],
    )
    plt.colorbar(im, label=cbar_label)
    plt.xlabel("Keq2 [-]")
    plt.ylabel("f1 HAME feed [mM/min]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_example_trajectory(model, test_df, example_sim, x_scaler, y_scaler, filename):
    true_traj = test_df[test_df["sim_id"] == example_sim].copy()

    x_example = true_traj[INPUT_COLS].values
    x_example_scaled = x_scaler.transform(x_example)
    x_example_t = torch.tensor(x_example_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_example_scaled = model(x_example_t).numpy()

    y_example = y_scaler.inverse_transform(y_example_scaled)
    time = true_traj["time_min"].values
    state_names = ["H", "hH", "oH", "aH", "dH", "Ala"]

    plt.figure(figsize=(14, 10))

    for i, state in enumerate(state_names):
        plt.subplot(3, 2, i + 1)
        plt.plot(time, true_traj[f"{state}_mM"].values, label="ODE")
        plt.plot(time, y_example[:, i], "--", label="NN")
        plt.title(state)
        plt.xlabel("Time [min]")
        plt.ylabel("Concentration [mM]")
        plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_all_final_heatmaps(final_cmp_df, output_dir):
    output_dir = Path(output_dir)
    output_states = ["H", "hH", "oH", "aH", "dH", "Ala"]

    for state in output_states:
        ode_grid = final_cmp_df.pivot(
            index="f1_mM_min",
            columns="Keq2_dimensionless",
            values=f"{state}_ode_mM",
        )
        nn_grid = final_cmp_df.pivot(
            index="f1_mM_min",
            columns="Keq2_dimensionless",
            values=f"{state}_nn_mM",
        )
        err_grid = final_cmp_df.pivot(
            index="f1_mM_min",
            columns="Keq2_dimensionless",
            values=f"{state}_error_mM",
        )

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        surfaces = [ode_grid, nn_grid, err_grid]
        titles = [f"{state} ODE", f"{state} NN", f"{state} NN − ODE"]

        vmax_err = np.max(np.abs(err_grid.values))

        for i, (z, title) in enumerate(zip(surfaces, titles)):
            ax = axes[i]

            kwargs = {
                "origin": "lower",
                "aspect": "auto",
                "extent": [
                    z.columns.min(),
                    z.columns.max(),
                    z.index.min(),
                    z.index.max(),
                ],
            }

            if i == 2:
                kwargs.update({"vmin": -vmax_err, "vmax": vmax_err})

            im = ax.imshow(z.values, **kwargs)
            ax.set_title(title)
            ax.set_xlabel("Keq2 [-]")
            ax.set_ylabel("f1 HAME feed [mM/min]")
            fig.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(output_dir / f"{state}_ODE_vs_NN_error.png", dpi=300)
        plt.close()
