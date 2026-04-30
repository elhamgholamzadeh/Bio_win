import numpy as np

STARTTIME = 0.0
STOPTIME = 480.0
N_TIMEPOINTS = 2000

T_EVAL = np.linspace(STARTTIME, STOPTIME, N_TIMEPOINTS)

INITIAL_ALA_M_M = 1.0

INITIAL_CONDITIONS = [
    0.0,              # H_mM
    0.0,              # hH_mM
    0.0,              # oH_mM
    0.0,              # aH_mM
    0.0,              # dH_mM
    INITIAL_ALA_M_M,  # Ala_mM
]

BASE_PARAMS = {
    "X_gCDW_L": 10.0,

    # Reaction 1: H -> hH
    "Vmax1_mmol_gCDW_min": 0.07,
    "Ks1_mM": 0.014,
    "Ki_H_1_mM": 16.7,

    # Reaction 2: hH <-> oH
    "Vr2_mmol_gCDW_min": 0.107,
    "Ks_oH2_mM": 0.02,

    # Reaction 3a: oH -> aH
    "Vmax3a_mmol_gCDW_min": 0.591,
    "Ks_oH3a_mM": 0.33,
    "Ks_Ala_mM": 0.06,

    # Reaction 3b: oH -> dH
    "Vmax3b_mmol_gCDW_min": 0.06,
    "Ks_oH3b_mM": 0.042,
    "Ki_H_3b_mM": 0.3,

    # Feeds
    "f1_mM_min": 0.1,
    "f2_mM_min": 1.0,

    # Feed stop time
    "t1_min": 720.0,
}

OUTPUT_COLS = [
    "H_mM",
    "hH_mM",
    "oH_mM",
    "aH_mM",
    "dH_mM",
    "Ala_mM",
]

INPUT_COLS = [
    "time_min",
    "Keq2_dimensionless",
    "f1_mM_min",
    "f2_mM_min",
    "Vr2_mmol_gCDW_min",
    "X_gCDW_L",
    "Vmax1_mmol_gCDW_min",
    "Ks1_mM",
    "Ki_H_1_mM",
    "Ks_oH2_mM",
    "Vmax3a_mmol_gCDW_min",
    "Ks_oH3a_mM",
    "Ks_Ala_mM",
    "Vmax3b_mmol_gCDW_min",
    "Ks_oH3b_mM",
    "Ki_H_3b_mM",
    "t1_min",
]
