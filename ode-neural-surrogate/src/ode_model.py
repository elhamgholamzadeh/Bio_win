from scipy.integrate import solve_ivp

from .config import (
    BASE_PARAMS,
    INITIAL_ALA_M_M,
    INITIAL_CONDITIONS,
    STARTTIME,
    STOPTIME,
    T_EVAL,
)


def odes(t, y, p):
    """ODE system for the biochemical reaction network."""
    H, hH, oH, aH, dH, Ala = y

    F_HAME = p["f1_mM_min"] if t <= p["t1_min"] else 0.0
    F_Ala = p["f2_mM_min"] if t <= p["t1_min"] else 0.0

    X = p["X_gCDW_L"]

    v1 = (
        p["Vmax1_mmol_gCDW_min"]
        * H
        / (p["Ks1_mM"] + H + (H ** 2) / p["Ki_H_1_mM"])
        * X
    )

    Vr2 = p["Vr2_mmol_gCDW_min"]
    Keq2 = p["Keq2_dimensionless"]
    Vf2 = Vr2 * Keq2

    v2 = ((Vf2 * hH - Vr2 * oH) / (p["Ks_oH2_mM"] + hH + oH) * X)

    v3a = (
        p["Vmax3a_mmol_gCDW_min"]
        * oH
        / (p["Ks_oH3a_mM"] + oH)
        * Ala
        / (p["Ks_Ala_mM"] + Ala)
        * X
    )

    v3b = (
        p["Vmax3b_mmol_gCDW_min"]
        * oH
        / (p["Ks_oH3b_mM"] + oH + (H ** 2) / p["Ki_H_3b_mM"])
        * X
    )

    v3 = v3a + v3b

    return [
        F_HAME - v1,
        v1 - v2,
        v2 - v3a - v3b,
        v3a,
        v3b,
        F_Ala - v3,
    ]


def simulate(keq2, f1, t_eval=T_EVAL, base_params=None):
    """Run one ODE simulation for a Keq2 and f1 pair."""
    params = (base_params or BASE_PARAMS).copy()
    params["Keq2_dimensionless"] = float(keq2)
    params["f1_mM_min"] = float(f1)

    sol = solve_ivp(
        lambda t, y: odes(t, y, params),
        (STARTTIME, STOPTIME),
        INITIAL_CONDITIONS,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )

    return sol, params
