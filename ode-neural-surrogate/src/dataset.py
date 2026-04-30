import numpy as np
import pandas as pd

from .config import INITIAL_ALA_M_M, T_EVAL
from .ode_model import simulate


def generate_dataset(
    keq2_values=None,
    f1_values=None,
    output_file=None,
):
    """Generate a full time-course dataset from ODE simulations."""
    if keq2_values is None:
        keq2_values = np.arange(0.01, 0.151, 0.01)
    if f1_values is None:
        f1_values = np.arange(0.02, 0.121, 0.01)

    all_rows = []
    sim_id = 0

    for keq2 in keq2_values:
        for f1 in f1_values:
            sim_id += 1

            sol, params = simulate(keq2=keq2, f1=f1, t_eval=T_EVAL)

            if not sol.success:
                print(f"Solver failed: Keq2={keq2}, f1={f1}")
                continue

            for i, time_min in enumerate(sol.t):
                H = sol.y[0, i]
                hH = sol.y[1, i]
                oH = sol.y[2, i]
                aH = sol.y[3, i]
                dH = sol.y[4, i]
                Ala = sol.y[5, i]

                total_analytes = H + hH + oH + aH + dH
                expected_HAME_feed = f1 * min(time_min, params["t1_min"])
                mass_balance_error = expected_HAME_feed - total_analytes

                expected_Ala = (
                    INITIAL_ALA_M_M
                    + params["f2_mM_min"] * min(time_min, params["t1_min"])
                    - aH
                    - dH
                )
                Ala_balance_error = expected_Ala - Ala

                row = {
                    "sim_id": sim_id,
                    "time_min": time_min,
                    "H_mM": H,
                    "hH_mM": hH,
                    "oH_mM": oH,
                    "aH_mM": aH,
                    "dH_mM": dH,
                    "Ala_mM": Ala,
                    "total_analytes_mM": total_analytes,
                    "expected_HAME_feed_mM": expected_HAME_feed,
                    "mass_balance_error_mM": mass_balance_error,
                    "expected_Ala_mM": expected_Ala,
                    "Ala_balance_error_mM": Ala_balance_error,
                }

                for key, value in params.items():
                    row[key] = value

                all_rows.append(row)

    dataset = pd.DataFrame(all_rows)

    if output_file:
        dataset.to_csv(output_file, index=False)

    print("Dataset generated")
    print("Total simulations:", sim_id)
    print("Total rows:", len(dataset))
    if output_file:
        print("Saved:", output_file)

    return dataset


def final_time_dataframe(dataset, output_file=None):
    """Return the final time point for each simulation."""
    final_df = (
        dataset.sort_values("time_min")
        .groupby("sim_id")
        .tail(1)
        .reset_index(drop=True)
    )
    if output_file:
        final_df.to_csv(output_file, index=False)
    return final_df


def add_toxicity_flag(final_df, limit_mM=2.7):
    """Flag final HAME concentrations above the toxicity threshold."""
    final_df = final_df.copy()
    final_df["H_above_toxic_limit"] = final_df["H_mM"] > limit_mM
    return final_df
