#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   calcFlux_HT8600.py
@Time       :   2024/07/15 16:03:54
@Author     :   Da Pan
@Contact    :   pd.phy.pku@gmail.com
@License    :   (C)Copyright 2022-2023, Da Pan
@Desc       :   None
"""

# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import ECTools
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
import pickle as pkl
from scipy.signal import butter, lfilter
from random import gauss
from scipy.odr import Model, Data, ODR
from scipy.stats import linregress
import warnings

warnings.filterwarnings("ignore")


# %% Define functions for spectroscopic corrections for H2O and pseudo mixing ratio
# =============================================================================
# These functions are used to calculate correct raw measurements and calculate
# for H2O and CH4 and calculate pseudo mixing ratio for CH4 measured by HT-8600P.
# =============================================================================
def cal_x_p(H2O, p, T, CH4):
    """
    Calculate pseudo mixing ratio of CH4 measured by HT-8600P

    :param H2O: mean-T corrected H2O molar density (mol+1m-3)
    :param p: pressure (hPa)
    :param T: Slow temperature (C)
    :param CH4: Raw CH4 mass density (ug+1m-3)
    """
    slow_T = np.nanmean(T) + 273.15  # K
    e = H2O * 18e-3 * R_w * slow_T / 1000  # kg/m3  # J/KG/K  # K  # kpa
    x_v = ECTools.molar_den2mixing(H2O, p / 10, slow_T, e)
    pe = p * (1 + alpha_v * x_v)
    slow_k = spec_correction(pe, slow_T) / spec_correction(1000, 299)  # hpa

    x_CH4_8600 = (
        ECTools.molar_den2mixing(
            CH4 / 16 * slow_k,  # umol
            p / 10,  # kpa
            slow_T,  # k
            e,  # kpa
        )
        * 16
        / 29
    )
    return x_CH4_8600


def cal_H2O_spec_den_corr_slow(H2O, p, T):
    """
    Calculate spectroscopic density correction for H2O with slow response time

    :param H2O: raw H2O molar density (mol+1m-3)
    :param p: pressure (hPa)
    :param T: Slow temperature (C)
    """
    slow_T = np.nanmean(T) + 273.15  # K
    e = H2O * 18e-3 * R_w * slow_T / 100  # kg/m3  # J/KG/K  # K  # hpa
    pe = p + alpha_v_H2O * e
    k = spec_correction_H2O(pe, slow_T) / spec_correction_H2O(1095, 299)  # hpa
    kp = spec_correction_H2O(pe + 1, slow_T) / spec_correction_H2O(1095, 299) - k  # hpa
    return H2O * (k - kp * alpha_v_H2O * e) / (1 - kp * alpha_v_H2O * e), k


def cal_H2O_spec_den_corr_fast(H2O, p, T):
    """
    Calculate spectroscopic density correction for H2O with fast response time

    :param H2O: raw H2O molar density (mol+1m-3)
    :param p: pressure (hPa)
    :param T: Fast temperature (K)
    """
    e = H2O * 18e-3 * R_w * T / 100  # kg/m3  # J/KG/K  # K  # hpa
    pe = p + alpha_v_H2O * e
    k = spec_correction_H2O(pe, T) / spec_correction_H2O(1095, 299)  # hpa
    kp = spec_correction_H2O(pe + 1, T) / spec_correction_H2O(1095, 299) - k  # hpa
    return H2O * (k - kp * alpha_v_H2O * e) / (1 - kp * alpha_v_H2O * e)


def cal_spec_den_corr(H2O, p, T, CH4):
    """
    Calculate spectroscopic density correction for CH4

    :param H2O: fast corrected H2O molar density (mol+1m-3)
    :param p: pressure (hPa)
    :param T: Fast temperature (K)
    """
    T = T + 273.15
    e = H2O * 18e-3 * R_w * T / 1000  # kg/m3  # J/KG/K  # K  # kpa
    x_v = ECTools.molar_den2mixing(H2O, p / 10, T, e)
    pe = p * (1 + alpha_v * x_v)
    k = spec_correction(pe, T) / spec_correction(1000, 299)  # hpa
    return CH4 * k


# %% Define constants
# =============================================================================
# These are constants for general calculation.
# =============================================================================
R_w = 461.5  # Ideal gas constant for water vapor, J/kg*K
R_d = 287.05  # Ideal gas constant for dry air, J/kg*K
R = 8.31  # Universal ideal gas constant, J/mol*K
L_v = 1000 * 2257  # Latent heat of vaporization (H2O), J/kg
C_p = 1005  # Approximate constant pressure specific heat of air, J/kg*K
k = 0.4  # Von Karman constant for EC calculation
g = 9.8  # Acceleration of gravity, m/s^2
gamma_d = 1.4  # Dry air specific heat ratio
alpha_v = 0.36  # 0.22#0.16  # CH4 foreign gas broadening coefficient for H2O density -1m 经过对比0.16/0.3/0.5的系数，发现.3的最接近0
alpha_v_H2O = 4.22

a = 17.271  # Dew point coefficient a
b = 237.7  # Dew point coefficient b
sonic_angle = 265  # Sonic heading angle
sonic_num = 1  # Number of sonic;
L_Kljun = np.genfromtxt("Kljun_L_95.csv", delimiter=",")
molec_weight = {"CH4": 0.017, "H2O": 0.018, "CO2": 0.044, "CH4": 0.044, "CH4": 0.016}

D_CH4 = 0.259
# %% Load spectroscopic params, wind rotation params
cf_file = "correction_factor_CH4_3104_new.csv"
df_cf = pd.read_csv(cf_file)
spec_correction = LinearNDInterpolator(
    np.c_[df_cf["p_grid"], df_cf["t_grid"]], df_cf["correction_factor"]
)
cf_file_h2o = "correction_factor_H2O_3106.csv"
df_cf_h2o = pd.read_csv(cf_file_h2o)
# df_cf_h2o["p_grid"]
spec_correction_H2O = LinearNDInterpolator(
    np.c_[df_cf_h2o["p_grid"], df_cf_h2o["t_grid"]], df_cf_h2o["correction_factor"]
)
C_mats, D_mats, offsets = pkl.load(open("planar_fit_params.pkl", mode="rb"))

# %% Load SCF from EddyPro output
eddy_path = "./HT8600EddyProOut/eddypro_0629_full_output_2025-12-10T112314_adv.csv"

eddy = pd.read_csv(
    eddy_path,
    skiprows=[0, 2],
    na_values=["-9999", "NaN", "nan"],
    parse_dates=[["date", "time"]],
).set_index("date_time")
eddy.index -= pd.Timedelta(minutes=30)

# %% Define site properties
# =============================================================================
# z is measurement height; d is displacement height. These parametes are
# specific for a measurement site.
# =============================================================================
z = 2.1
d = 0.2
rough_len = 0.07
f_coriolis = 2 * 7.2921e-5 * np.sin(40.3 / 180 * 3.14)
lagOffset = -2

# %% Define data directory and will use
# =============================================================================
# data_dir: Upper directory that holds 2022 data
# file_list: all data files
# =============================================================================
# data_dir = "./HT8600/"
data_dir = "./1_half_hour/half_hour/"
file_list = sorted(os.listdir(data_dir))

# %% Read data
# =============================================================================
# Loop through each file. Timestamp from filename is put into a list.
# Only WT is calculated. Triple rotation will be used temporarily.
# Cospectrum is also calculated.
# =============================================================================
f_wt_list = []
cosp_wt_list = []
f_wCH4_list = []
cosp_wCH4_list = []
f_wh2o_list = []
cosp_wh2o_list = []
f_wco2_list = []
cosp_wco2_list = []
results = pd.DataFrame(
    {
        "T": [],
        "rho_air": [],
        "e": [],
        "theta_v": [],
        "p": [],
        "WS": [],
        "WD": [],
        "e": [],
        "es": [],
        "CH4": [],
        "x_CH4": [],
        "F_CH4": [],
        "F_CH4_b": [],
        "mean_cf": [],
        "CO2": [],
        "x_CO2": [],
        "F_CO2": [],
        "H2O": [],
        "x_H2O": [],
        "F_H2O": [],
        "cov_wk": [],
        "cov_wkc": [],
        "cov_uu": [],
        "cov_ww": [],
        "cov_tt": [],
        "cov_CH4CH4": [],
        "cov_h2oh2o": [],
        "cov_wt": [],
        "cov_wh2o": [],
        "cov_wco2": [],
        "cov_wCH4": [],
        "u*": [],
        "L": [],
        "scf_CH4": [],
        "scf_co2": [],
        "scf_h2o": [],
        "wpl_h2o": [],
        "wpl_co2": [],
        "wpl_CH4": [],
        "itc_CH4": [],
        "itc_t": [],
        "itc_w": [],
        "itc_u": [],
        "itc_h2o": [],
        "itc_co2": [],
        "sst_wt": [],
        "sst_wCH4": [],
        "sst_wco2": [],
        "sst_wh2o": [],
        "err_wCH4_instr": [],
        "err_wh2o_instr": [],
        "err_wco2_instr": [],
        "err_wCH4_stoch": [],
        "err_wh2o_stoch": [],
        "err_wco2_stoch": [],
        "err_wCH4_salsky": [],
        "err_wh2o_salsky": [],
        "err_wco2_salsky": [],
        "x_70%": [],
        "S": [],
        "S_corrected": [],
        "na_t": [],
        "na_w": [],
        "na_co2": [],
        "na_h2o": [],
        "na_CH4": [],
        "na_t_despike": [],
        "na_w_despike": [],
        "na_co2_despike": [],
        "na_CH4_despike": [],
        "na_h2o_despike": [],
    }
)

for filename in file_list[50:60]:
    if "CR6" in filename:
        # Print the name of the file being processed
        print(filename)
        # Load timestamp from filename
        TS = pd.to_datetime(filename[-20:-4], format="%Y-%m-%d_%H_%M")
        # Read 10 Hz data
        data_10hz = ECTools.read_TOA5(
            data_dir + filename,
            time_col="TIMESTAMP",
        )
        # Remove potential duplicate indices
        data_10hz = data_10hz.loc[TS:]
        data_10hz["p"] = data_10hz["PA"]

        # =====================================================================
        # Pre-processing for gas concentrations
        # =====================================================================
        # Convert measured H2O density to molar density
        data_10hz["H2O"] = data_10hz["H2O_density_8600"] / 18
        # Calculate slow-response spectroscopic correction for H2O
        data_10hz["H2O_slow"], data_10hz["k_v_slow"] = cal_H2O_spec_den_corr_slow(
            data_10hz["H2O"], data_10hz["Pressure_8600"], data_10hz["amb_tmpr_77"]
        )
        # Save raw H2O molar density
        data_10hz["H2O_raw"] = data_10hz["H2O_density_8600"] / 18
        # Calculate water vapor pressure and mole fraction for slow and raw H2O
        data_10hz["e_slow"] = (
            data_10hz["H2O_slow"]
            * 18e-3
            * R_w
            * (data_10hz["amb_tmpr_77"] + 273.15)
            / 100
        )
        data_10hz["x_H2O_slow"] = data_10hz["e_slow"] / (
            data_10hz["Pressure_8600"] - data_10hz["e_slow"]
        )
        data_10hz["e_raw"] = (
            data_10hz["H2O_raw"]
            * 18e-3
            * R_w
            * (data_10hz["amb_tmpr_77"] + 273.15)
            / 100
        )
        # Calculate spectroscopic correction factor for CH4 using slow H2O
        pe_slow = data_10hz["Pressure_8600"] + alpha_v * data_10hz["e_slow"]
        pe_raw = data_10hz["Pressure_8600"] + alpha_v * data_10hz["e_raw"]
        data_10hz["k_slow"] = spec_correction(
            pe_slow, np.mean(data_10hz["amb_tmpr_77"] + 273.15)
        ) / spec_correction(1000, 299)

        # Convert mass densities to molar densities and calculate pseudo mixing ratio
        data_10hz["CO2"] = data_10hz["CO2_density"] / 44
        data_10hz["CH4"] = data_10hz["CH4_density_8600"] / 16  # umol
        data_10hz["raw_CH4"] = data_10hz["CH4_density_8600"] / 16
        data_10hz["Xmr_CH4"] = cal_x_p(
            data_10hz["H2O_slow"],
            data_10hz["p"] * 10,
            # data_10hz["Pressure_8600"],
            data_10hz["amb_tmpr_77"],
            # data_10hz["T_SONIC"],
            data_10hz["CH4_density_8600"],
        )  # ug/g

        # =====================================================================
        # Check data availability
        # ====================================================================
        na_t = np.sum(data_10hz["T_SONIC"].isnull())
        na_w = np.sum(data_10hz["Uz"].isnull())
        na_co2 = np.sum(data_10hz["H2O"].isnull())
        na_CH4 = np.sum(data_10hz["CH4"].isnull())

        if na_CH4 < 1800:  # more than 90% data available
            # =====================================================================
            # Get SCF from EddyPro output
            # ====================================================================
            if TS in eddy.index:
                scf_h2o = eddy.loc[TS, "h2o_scf"]
                scf_ch4 = eddy.loc[TS, "ch4_scf"]
                scf_csat3 = eddy.loc[TS, "H_scf"]
                scf_7500 = eddy.loc[TS, "co2_scf"]
            else:
                scf_h2o = 1.0
                scf_ch4 = 1.0
                scf_csat3 = 1.0
                scf_7500 = 1.0

            # =====================================================================
            # Rotate wind.
            # =====================================================================
            data_10hz["u"] = data_10hz["Ux"]
            data_10hz["v"] = data_10hz["Uy"]
            data_10hz["w"] = data_10hz["Uz"]

            WD = np.angle(np.mean(data_10hz["Ux"]) + np.mean(data_10hz["Uy"]) * 1j)
            np.degrees(WD) - 110

            Ux_mean = np.mean(data_10hz["Ux"])  # North-south component
            Uy_mean = np.mean(data_10hz["Uy"])  # East-west component

            WD_deg = (270 - np.degrees(np.arctan2(Uy_mean, Ux_mean))) % 360
            (270 - 50) % 360

            # Calculate wind sector and optain planar fit rotation matrix
            # wind_sect = int(np.floor(((np.rad2deg(WD) + 95) % 360) / 45))
            # P_mat = np.matmul(C_mats[wind_sect], D_mats[wind_sect]).transpose()
            # ECTools.planar_fit_wind_rotata(data_10hz, P_mat, offsets[wind_sect], WD)

            # In this case, use triple rotation
            data_10hz["u_rot"], data_10hz["v_rot"], data_10hz["w_rot"], _, _ = (
                ECTools.rotate_wind_vector(
                    data_10hz["u"], data_10hz["v"], data_10hz["w"], method="TR", k=1
                )
            )
            WS = np.mean((data_10hz["u_rot"]))

            # ====================================================================
            # Despike and time lag correction
            # =====================================================================
            ECTools.despike(
                data_10hz,
                [
                    "CH4",
                    "Xmr_CH4",
                    "u_rot",
                    "v_rot",
                    "w_rot",
                    "T_SONIC",
                    "CO2",
                    "H2O",
                    "raw_CH4",
                ],
                lim=[7],
                method="MAD",
            )
            # Use cross-covariance maxmization to determine time lags
            ECTools.shift_lag_max(
                data_10hz,
                ["H2O", "Xmr_CH4", "CH4", "raw_CH4", "CO2"],
                "w_rot",
                lag_lim=5,
            )

            # =====================================================================
            # Correct T_sonic and calculate air properties
            # =====================================================================

            data_10hz["T_SONIC"] += 273.15
            (
                air_prop,
                data_10hz["theta_v"],
                data_10hz["q"],
                data_10hz["T"],
                data_10hz["e"],
            ) = ECTools.calc_air_properties(
                data_10hz["T_SONIC"],
                data_10hz["H2O"],
                data_10hz["p"],
            )

            # =====================================================================
            # Correct H2O and CH4 spectroscopic effects, calculate mixing ratios
            # =====================================================================
            mean_T = np.nanmean(data_10hz["T"])
            # Correct H2O molar density with fast response spectroscopic correction
            data_10hz["H2O"] = cal_H2O_spec_den_corr_fast(
                data_10hz["H2O"],
                data_10hz["p"] * 10,
                (data_10hz["T"] - mean_T) / scf_h2o * scf_csat3 + mean_T,
            )
            # Re-correct sonic temperature and recalculate air properties using
            # corrected H2O
            (
                air_prop,
                data_10hz["theta_v"],
                data_10hz["q"],
                data_10hz["T"],
                data_10hz["e"],
            ) = ECTools.calc_air_properties(
                data_10hz["T_SONIC"],
                data_10hz["H2O"],
                data_10hz["p"],
            )
            # Calculate mixing ratios for H2O and CO2
            data_10hz["x_H2O"] = ECTools.molar_den2mixing(
                data_10hz["H2O"], data_10hz["p"], data_10hz["T"], data_10hz["e"]
            )

            data_10hz["x_CO2"] = ECTools.molar_den2mixing(
                data_10hz["CO2"], data_10hz["p"], data_10hz["T"], data_10hz["e"]
            )
            # Correct CH4 mass density with spectroscopic correction
            p_e = data_10hz["p"] * (1 + alpha_v * data_10hz["x_H2O"]) * 10
            cf = spec_correction(
                p_e.values, (data_10hz["T"] - mean_T) / scf_ch4 * scf_csat3 + mean_T
            ) / spec_correction(1000, 299)
            data_10hz["CH4_K_T"] = data_10hz["CH4"] * cf
            data_10hz["cf"] = cf
            data_10hz["k_slow_diff"] = data_10hz["cf"] - data_10hz["k_slow"]
            data_10hz["k_relative_diff"] = (
                data_10hz["k_slow_diff"] / data_10hz["k_slow"]
            )
            data_10hz["CH4"] *= cf
            avg_theta_v = np.mean(data_10hz["theta_v"])
            data_10hz["x_CH4"] = ECTools.molar_den2mixing(
                data_10hz["CH4"], data_10hz["p"], data_10hz["T"], data_10hz["e"]
            )
            # Obtain mean gas densities for flux calculation
            gas_den_avg = ECTools.calc_avg(
                data_10hz,
                [
                    "CH4",
                    "H2O",
                    "CO2",
                    "Xmr_CH4",
                    "raw_CH4",
                    "v_rot",
                    "u_rot",
                    "w_rot",
                    "T",
                    "H2O_slow",
                    "x_H2O_slow",
                    "e_slow",
                    "k_slow",
                    "H2O_raw",
                    "CH4_density_8600",
                ],
            )
            # ====================================================================
            # Calculate X(T, rho_vm), chi_c (corrected from x_p), MT, Mv, MT_x (XT)
            # Mv_x (Xv) for flux calculation
            # =====================================================================
            # Calculate X(T, rho_vm) correction for Xmr_CH4 (X_p)
            coefficient = (
                data_10hz["cf"]
                / data_10hz["k_slow"]
                * (
                    (data_10hz["T"] - np.nanmean(data_10hz["T"])) / scf_ch4 * scf_csat3
                    + np.nanmean(data_10hz["T"])
                )
                / np.nanmean(data_10hz["amb_tmpr_77"] + 273.15)
                * (data_10hz["p"] - data_10hz["e_slow"] / 10)
                / (data_10hz["p"] - data_10hz["e"])
            )
            # Correct chi_p to chi_c
            data_10hz["Xmr_CH4_coed"] = data_10hz["Xmr_CH4"] * coefficient

            # calculate kT, kv for H2O and CH4
            xh2o1 = 1 / (
                data_10hz["Pressure_8600"]
                * 1e2
                / data_10hz["H2O_raw"]
                / 8.314
                / (data_10hz["amb_tmpr_77"] + 273.15)
                - 1
            )
            xh2o2 = 1 / (
                data_10hz["Pressure_8600"]
                * 1e2
                / (data_10hz["H2O_raw"] + 0.05)
                / 8.314
                / (data_10hz["amb_tmpr_77"] + 273.15)
                - 1
            )
            xh2o3 = 1 / (
                data_10hz["Pressure_8600"]
                * 1e2
                / data_10hz["H2O_raw"]
                / 8.314
                / (data_10hz["amb_tmpr_77"] + 274.15)
                - 1
            )
            amb_T = np.nanmean(data_10hz["amb_tmpr_77"] + 273.15)
            p_e1_v = data_10hz["p"] * (1 + alpha_v_H2O * xh2o1) * 10
            p_e2_v = data_10hz["p"] * (1 + alpha_v_H2O * xh2o2) * 10
            p_e3_v = data_10hz["p"] * (1 + alpha_v_H2O * xh2o3) * 10
            k_v = spec_correction_H2O(
                np.nanmean(p_e1_v),
                amb_T,
            ) / spec_correction_H2O(1095, 299)
            k1_v = spec_correction_H2O(
                np.nanmean(p_e2_v),
                amb_T,
            ) / spec_correction_H2O(1095, 299)
            k2_v = spec_correction_H2O(
                np.nanmean(p_e3_v),
                amb_T + 1,
            ) / spec_correction_H2O(1095, 299)
            k3_v = spec_correction_H2O(
                np.nanmean(p_e1_v), amb_T + 1
            ) / spec_correction_H2O(1095, 299)
            kv_v = (k1_v - k_v) / 0.05 / 0.018
            kt_v = (k2_v - k_v) / 1
            kt_v2 = (k3_v - k_v) / 1
            xh2o1 = 1 / (
                data_10hz["p"] * 1e3 / data_10hz["H2O_raw"] / 8.314 / (data_10hz["T"])
                - 1
            )
            xh2o2 = 1 / (
                data_10hz["p"]
                * 1e3
                / (data_10hz["H2O_raw"] + 0.05)
                / 8.314
                / (data_10hz["T"])
                - 1
            )
            xh2o3 = 1 / (
                data_10hz["p"]
                * 1e3
                / data_10hz["H2O_raw"]
                / 8.314
                / (data_10hz["T"] + 1)
                - 1
            )
            p_e1_c = data_10hz["p"] * (1 + alpha_v * xh2o1) * 10
            p_e2_c = data_10hz["p"] * (1 + alpha_v * xh2o2) * 10
            p_e3_c = data_10hz["p"] * (1 + alpha_v * xh2o3) * 10
            k = spec_correction(
                np.nanmean(p_e1_c), air_prop["T_avg"]
            ) / spec_correction(1000, 299)
            k1 = spec_correction(
                np.nanmean(p_e2_c), air_prop["T_avg"]
            ) / spec_correction(1000, 299)
            k2 = spec_correction(
                np.nanmean(p_e3_c), air_prop["T_avg"] + 1
            ) / spec_correction(1000, 299)
            kv = (k1 - k) / 0.05 / 0.018
            kt = (k2 - k) / 1
            # Calculate MT and Mv using kt_c, kv_c
            MT = (
                np.nanmean(data_10hz["cf"])
                + air_prop["T_avg"]
                / (1 + 1.6077 * gas_den_avg["H2O"] * 0.018 / air_prop["rho_air"])
                * kt
            )
            Mv = np.nanmean(data_10hz["cf"]) + (air_prop["rho_air"]) / 1.6077 * kv
            # Calculate MT_x and Mv_x using dX_dT and dX_dv
            K_v = spec_correction_H2O(np.nanmean(p_e1_v), amb_T) / spec_correction_H2O(
                1095, 299
            )
            K_v2 = spec_correction_H2O(
                np.nanmean(p_e1_v), amb_T + 1
            ) / spec_correction_H2O(1095, 299)
            K_v3 = spec_correction_H2O(
                np.nanmean(p_e1_v) + 1, amb_T
            ) / spec_correction_H2O(1095, 299)
            Kt_v = K_v2 - K_v
            Kp_v = (K_v3 - K_v) / 100
            K_c = spec_correction(
                np.nanmean(p_e1_c), air_prop["T_avg"]
            ) / spec_correction(1000, 299)
            K_c2 = spec_correction(
                np.nanmean(p_e1_c), air_prop["T_avg"] + 1
            ) / spec_correction(1000, 299)
            K_c3 = spec_correction(
                np.nanmean(p_e1_c) + 1, air_prop["T_avg"]
            ) / spec_correction(1000, 299)
            Kt_c = K_c2 - K_c
            Kp_c = (K_c3 - K_c) / 100
            dX_dT, dX_dv = ECTools.calculate_X_partials_rhovm(
                air_prop["T_avg"],
                amb_T,
                np.nanmean(data_10hz["Pressure_8600"]),
                gas_den_avg["H2O_raw"] * 0.018,
                gas_den_avg["H2O"] * 0.018,
                gas_den_avg["H2O_slow"] * 0.018,
                K_v,
                np.nanmean(data_10hz["k_v_slow"]),
                np.nanmean(data_10hz["cf"]),
                K_c,
                Kp_v,
                Kt_v,
                Kp_c,
                Kt_c,
                8.31 / 0.018 / 100,
                alpha_v_H2O,
                alpha_v,
            )
            X = np.nanmean(coefficient)
            MT_x = dX_dT * air_prop["T_avg"]
            Mv_x = dX_dv

            # =====================================================================
            # Detrend data.
            # =====================================================================
            data_detrend = ECTools.detrend(
                data_10hz,
                [
                    "u_rot",
                    "v_rot",
                    "w_rot",
                    "T",
                    "CO2",
                    "CH4",
                    "Xmr_CH4",
                    "H2O",
                    "raw_CH4",
                    "CH4_K_T",
                    "Xmr_CH4_coed",
                    "H2O_slow",
                    # "dX_dT",
                    # "dX_dv",
                ],
                method="block_average",
            )

            # =====================================================================
            # Calculate cospectra (disabled because EddyPro is used for HFA)
            # =====================================================================
            cov5_list = []

            f_list = []
            cov_list = []
            cosp_list = []
            co_list = []
            col_pairs = [
                ["w_rot", "T"],
                ["w_rot", "CO2"],
                ["w_rot", "H2O"],
                ["w_rot", "CH4"],
                # ["w_rot", "Xmr_CH4"],
            ]
            na_t_despike = np.sum(data_10hz["T"].isnull())
            na_w_despike = np.sum(data_10hz["w_rot"].isnull())
            na_co2_despike = np.sum(data_10hz["H2O"].isnull())
            na_CH4_despike = np.sum(data_10hz["CH4"].isnull())
            na_h2o_despike = np.sum(data_10hz["H2O"].isnull())

            # =====================================================================
            # Calculate raw covariance
            # =====================================================================
            cov30 = np.cov(
                data_detrend[
                    [
                        "u_rot",
                        "v_rot",
                        "w_rot",  # 2
                        "T",
                        "H2O",
                        "CO2",
                        "CH4",  # 6
                        "theta_v",
                        "Xmr_CH4",  # 8
                        "cf",
                        "raw_CH4",  # 10
                        "CH4_K_T",  # 11
                        "Xmr_CH4_coed",  # 12
                        "H2O_raw",  # 13
                    ]
                ]
                .interpolate()
                .dropna(how="any")
                .transpose()
            )
            # =====================================================================
            # Calculate measurement error
            # =====================================================================
            # Apply high-pass filter to data following Mauder et al. (2013)
            nyq = 0.5 * 10
            high = 1 / 7.5 / 60 / nyq
            low = 1.5e-2 / nyq
            b, a = butter(4, high, "highpass")
            f_data = (
                data_detrend[["CH4", "w_rot", "H2O", "CO2", "Xmr_CH4", "T", "raw_CH4"]]
                .interpolate()
                .fillna(0)
            )
            f_data["CH4"] = lfilter(b, a, f_data["CH4"])
            f_data["w_rot"] = lfilter(b, a, f_data["w_rot"])
            f_data["H2O"] = lfilter(b, a, f_data["H2O"])
            f_data["CO2"] = lfilter(b, a, f_data["CO2"])
            f_data["Xmr_CH4"] = lfilter(b, a, f_data["Xmr_CH4"])
            f_data["T"] = lfilter(b, a, f_data["T"])
            # corrected density CH4 flux errors
            autocov_list = []
            for p in range(10, 50, 10):
                autocov_list.append(ECTools.autocov(f_data, "w_rot", p))
            p_offset = np.polyfit(range(10, 50, 10), autocov_list, 1)[1]

            autocov_list = []
            for p in range(10, 50, 10):
                autocov_list.append(ECTools.autocov(f_data, "CH4", p))
            p_offset = np.polyfit(range(10, 50, 10), autocov_list, 1)[1]
            err_CH4_instr = (
                np.sqrt(np.abs((cov30[6, 6] - p_offset)) * cov30[2, 2] / 18000)
                / cov30[2, 6]
            )
            err_CH4_instr_absorbe = np.sqrt(
                np.abs(
                    ((cov30[6, 6] - p_offset)) * cov30[2, 2]
                    + (cov30[2, 2] - p_offset) * cov30[6, 6]
                )
                / 18000
            )
            err_CH4_stoch_FS = ECTools.stoch_err_FS(f_data, "CH4") / cov30[2, 6]

            # Raw CH4 flux errors
            autocov_list = []
            for p in range(10, 50, 10):
                autocov_list.append(ECTools.autocov(f_data, "raw_CH4", p))
            p_offset = np.polyfit(range(10, 50, 10), autocov_list, 1)[1]
            err_rawCH4_instr = (
                np.sqrt(np.abs((cov30[10, 10] - p_offset)) * cov30[2, 2] / 18000)
                / cov30[2, 10]
            )
            err_rawCH4_stoch_FS = ECTools.stoch_err_FS(f_data, "raw_CH4") / cov30[2, 10]

            # x_p flux errors
            autocov_list = []
            for p in range(10, 50, 10):
                autocov_list.append(ECTools.autocov(f_data, "Xmr_CH4", p))
            p_offset = np.polyfit(range(10, 50, 10), autocov_list, 1)[1]
            err_xmr_instr = (
                np.sqrt(np.abs((cov30[8, 8] - p_offset)) * cov30[2, 2] / 18000)
                / cov30[2, 8]
            )
            err_xmr_stoch_FS = ECTools.stoch_err_FS(f_data, "Xmr_CH4") / cov30[2, 8]

            # CO2 flux errors
            autocov_list = []
            for p in range(10, 50, 10):
                autocov_list.append(ECTools.autocov(f_data, "CO2", p))
            p_offset = np.polyfit(range(10, 50, 10), autocov_list, 1)[1]
            err_co2_instr = (
                np.sqrt(np.abs((cov30[5, 5] - p_offset)) * cov30[2, 2] / 18000)
                / cov30[2, 5]
            )
            err_co2_stoch_FS = ECTools.stoch_err_FS(f_data, "CO2") / cov30[2, 5]

            # H2O flux errors
            autocov_list = []
            for p in range(10, 50, 10):
                autocov_list.append(ECTools.autocov(f_data, "H2O", p))
            p_offset = np.polyfit(range(10, 50, 10), autocov_list, 1)[1]
            err_h2o_instr = (
                np.sqrt(np.abs((cov30[4, 4] - p_offset)) * cov30[2, 2] / 18000)
                / cov30[2, 4]
            )
            err_h2o_stoch_FS = ECTools.stoch_err_FS(f_data, "H2O") / cov30[2, 4]

            # Temperature flux errors
            autocov_list = []
            for p in range(10, 50, 10):
                autocov_list.append(ECTools.autocov(f_data, "T", p))
            p_offset = np.polyfit(range(10, 50, 10), autocov_list, 1)[1]
            err_T_instr = (
                np.sqrt(np.abs((cov30[3, 3] - p_offset)) * cov30[2, 2] / 18000)
                / cov30[2, 3]
            )
            err_T_stoch_FS = ECTools.stoch_err_FS(f_data, "T") / cov30[2, 3]

            # =====================================================================
            # Correct for high and low frequency loss (disabled because EddyPro is used for HFA)
            # =====================================================================
            # f = np.logspace(-3, 1, 65)
            # p = 0.1
            # s_licor = np.array([0.02, -0.25])
            # s_CH4 = np.array([0.03, 0.30])
            # hd = np.deg2rad(0) + WD
            # rotate_matix = np.array(
            #     [[np.cos(hd), -np.sin(hd)], [np.sin(hd), np.cos(hd)]]
            # )
            # s_licor = np.abs(np.matmul(rotate_matix.transpose(), s_licor))
            # s_CH4 = np.abs(np.matmul(rotate_matix.T, s_CH4))
            # n_p_CH4 = p * f / WS
            # n_s_licor_lat = s_licor[0] * f / WS
            # n_s_CH4_lat = s_CH4[0] * f / WS
            # n_s_licor_lon = s_licor[1] * f / WS
            # n_s_CH4_lon = s_CH4[1] * f / WS
            # n_p_w = 0.12 * f / WS
            # n_p_licor = 0.125 * f / WS
            # n = f * (z - d) / WS
            # T_w_sonic = ECTools.T_w(n_p_w)
            # T_p_sonic = ECTools.T_p_vertical(n_p_w)
            # T_p_licor = ECTools.T_p_vertical(n_p_licor)
            # T_p_CH4 = ECTools.T_p_vertical(n_p_CH4)
            # T_s_licor_hor = ECTools.T_s(n_s_licor_lat)
            # T_s_licor_ver = ECTools.T_s(n_s_licor_lon)
            # T_s_CH4_hor = ECTools.T_s(n_s_CH4_lat)
            # T_s_CH4_ver = ECTools.T_s(n_s_CH4_lon)
            last_cov30 = cov30
            # iter_n = 0
            # flag = True
            # while flag:
            # u_star = (last_cov30[0, 2] ** 2 + last_cov30[1, 2] ** 2) ** 0.25
            # L_ob = -(u_star**3.0) / last_cov30[2, 7] / k / g * avg_theta_v
            # xi = (z - d) / L_ob
            # Sww_m = n * ECTools.model_spectral_ww(n, xi, z, d)
            # Suu_m = n * ECTools.model_spectral_uu(n, xi, WS, f)
            # Swu_m = n * ECTools.model_spectral_wu(n, xi)
            # Sss_m = n * ECTools.model_spectral_ss(n, xi)
            # Sws_m = n * ECTools.model_spectral_ws(n, xi) (空间误差)
            # scf_ww = integrate.romb(Sww_m * T_w_sonic, 4.0 / 64) / integrate.romb(
            #     Sww_m, 4.0 / 64
            # )
            # scf_uu = integrate.romb(Suu_m * T_w_sonic, 4.0 / 64) / integrate.romb(
            #     Suu_m, 4.0 / 64
            # )
            # scf_wu = integrate.romb(Swu_m * T_w_sonic, 4.0 / 64) / integrate.romb(
            #     Swu_m, 4.0 / 64
            # )
            # scf_wt = integrate.romb(
            #     Sws_m * np.sqrt(T_w_sonic), 4.0 / 64
            # ) / integrate.romb(Sws_m, 4.0 / 64)
            # scf_wq = integrate.romb(
            #     Sws_m
            #     * np.sqrt(T_w_sonic * T_p_licor)
            #     * T_s_licor_ver
            #     * T_s_licor_hor,
            #     4.0 / 64,
            # ) / integrate.romb(Sws_m, 4.0 / 64)
            # scf_qq = integrate.romb(Sss_m * T_p_licor, 4.0 / 64) / integrate.romb(
            #     Sss_m, 4.0 / 64
            # )
            # scf_CH4CH4 = integrate.romb(Sss_m * T_p_CH4, 4.0 / 64) / integrate.romb(
            #     Sss_m, 4.0 / 64
            # )
            # scf_wCH4 = integrate.romb(
            #     Sws_m * np.sqrt(T_p_CH4 * T_w_sonic) * T_s_CH4_ver * T_s_CH4_hor,
            #     4.0 / 64,
            # ) / integrate.romb(Sws_m, 4.0 / 64)
            # scf_mat = np.array(
            #     [
            #         [
            #             1.0 / scf_uu,
            #             1.0 / scf_uu,
            #             1.0 / scf_wu,
            #             1.0,
            #             1.0,
            #             1.0,
            #             1.0,
            #             1.0,
            #             1.0,
            #             1.0,
            #         ],
            #         [
            #             1.0 / scf_uu,
            #             1.0 / scf_uu,
            #             1.0 / scf_wu,
            #             1.0,
            #             1.0,
            #             1.0,
            #             1.0,
            #             1.0,
            #             1.0,
            #             1.0,
            #         ],
            #         [
            #             1.0 / scf_wu,
            #             1.0 / scf_wu,
            #             1.0 / scf_ww,
            #             1.0 / scf_wt,
            #             1.0 / scf_wq,
            #             1.0 / scf_wq,
            #             1.0 / scf_wCH4,
            #             1.0 / scf_wt,
            #             1.0 / scf_wCH4,
            #             1.0,
            #         ],
            #         [1.0, 1.0, 1.0 / scf_wt, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            #         [
            #             1.0,
            #             1.0,
            #             1.0 / scf_wq,
            #             1.0,
            #             1.0 / scf_qq,
            #             1.0,
            #             1.0,
            #             1.0,
            #             1.0,
            #             1.0,
            #         ],
            #         [
            #             1.0,
            #             1.0,
            #             1.0 / scf_wq,
            #             1.0,
            #             1.0,
            #             1.0 / scf_qq,
            #             1.0,
            #             1.0,
            #             1.0,
            #             1.0,
            #         ],
            #         [
            #             1.0,
            #             1.0,
            #             1.0 / scf_wCH4,
            #             1.0,
            #             1.0,
            #             1.0,
            #             1.0 / scf_CH4CH4,
            #             1.0,
            #             1.0 / scf_CH4CH4,
            #             1.0,
            #         ],
            #         [1.0, 1.0, 1.0 / scf_wt, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            #         [
            #             1.0,
            #             1.0,
            #             1.0 / scf_wCH4,
            #             1.0,
            #             1.0,
            #             1.0,
            #             1.0 / scf_CH4CH4,
            #             1.0,
            #             1.0 / scf_CH4CH4,
            #             1.0,
            #         ],
            #         [1.0] * 10,
            #     ]
            # )
            # diff_mat = (cov30 * scf_mat - last_cov30) / last_cov30
            # last_cov30 = cov30 * scf_mat
            # iter_n += 1
            # if ((diff_mat[2, 3] < 1e-4) & (diff_mat[2, 4] < 1e-4)) | (iter_n > 10):
            #     flag = False

            # =====================================================================
            # Calculate u*, itc, and sst
            # =====================================================================
            cov30 = last_cov30
            u_star = (last_cov30[0, 2] ** 2 + last_cov30[1, 2] ** 2) ** 0.25
            kk = 0.4
            L_ob = -(u_star**3.0) / last_cov30[2, 7] / kk / g * avg_theta_v
            xi = (z - d) / L_ob
            itc_w = ECTools.itc_test_w(
                L_ob, d, z, np.var(data_detrend["w_rot"]), u_star, f_coriolis
            )
            itc_t = ECTools.itc_test_t(
                L_ob,
                d,
                z,
                np.var(data_detrend["T"]),
                np.abs(cov30[2, 3] / u_star),
                f_coriolis,
            )
            itc_CH4 = ECTools.itc_test_t(
                L_ob,
                d,
                z,
                np.var(data_detrend["CH4"]),
                np.abs(cov30[2, 6] / u_star),
                f_coriolis,
            )
            itc_u = ECTools.itc_test_u(
                L_ob, d, z, np.var(data_detrend["u_rot"]), u_star, f_coriolis
            )
            itc_co2 = ECTools.itc_test_t(
                L_ob,
                d,
                z,
                np.var(data_detrend["CO2"]),
                np.abs(cov30[2, 5] / u_star),
                f_coriolis,
            )
            itc_h2o = ECTools.itc_test_t(
                L_ob,
                d,
                z,
                np.var(data_detrend["H2O"]),
                np.abs(cov30[2, 4] / u_star),
                f_coriolis,
            )
            sst = ECTools.stationarity_test(
                data_detrend,
                [["w_rot", "T"], ["w_rot", "CH4"], ["w_rot", "CO2"], ["w_rot", "H2O"]],
            )
            # =====================================================================
            # Calculate Sensible, Latent, and CO2 fluxes and apply HFA corrections
            # using SCFs from EddyPro
            # =====================================================================

            E0 = cov30[2, 4] * 0.018 * scf_h2o
            # Calculate sensible heat flux
            S = cov30[2, 3] * C_p * air_prop["rho_air"]
            # Apply SCF correction
            S_corrected = S * scf_csat3
            # Calculate latent heat flux with WPL correction
            E = ECTools.WPL_H2O(
                E0,
                S_corrected,
                air_prop["rho_air"],
                gas_den_avg["H2O"] * 0.018,
                air_prop["T_avg"],
            )  # kg/m2s
            # Calculate CO2 flux with WPL and HFA correction
            F_CO2 = (
                ECTools.WPL_gas(
                    E,
                    air_prop["rho_air"],
                    gas_den_avg["H2O"] * 0.018,
                    S_corrected,
                    air_prop["T_avg"],
                    gas_den_avg["CO2"] * 44e-6,
                    cov30[2, 5] * 44e-6 * scf_7500,
                )
                * 1e3
            )
            # Raw x_p flux calculation
            F_CH4_x_p_raw = cov30[2, 8] * air_prop["rho_air"] * scf_ch4  # mg/(m^2 s)
            # Flux calculated using x_c
            F_CH4_x_c = cov30[2, 12] * air_prop["rho_air"] * scf_ch4  # mg/(m^2 s)
            # Calculate flux using x_p with full correction
            F_CH4_x_p_coed = (
                ECTools.mix_correction(
                    air_prop["rho_air"],
                    gas_den_avg["H2O"] * 0.018,
                    S_corrected,
                    air_prop["T_avg"],
                    gas_den_avg["Xmr_CH4"] * 1e-6,
                    F_CH4_x_p_raw * 1e-6,
                    MT_x,
                    cov30[2, 13] * 0.018 * scf_h2o,
                    Mv_x,
                )  # ng/m2/s
                * 1e6
            )  # mg/m2/s

            # CH4 flux calculated with H2O-related corrections and HFA corrections,
            # using point-by-point corrected CH4 mass density
            F_CH4_w = (
                ECTools.WPL_gas(
                    E,
                    air_prop["rho_air"],
                    gas_den_avg["H2O"] * 0.018,
                    0.0,
                    air_prop["T_avg"],
                    gas_den_avg["CH4"] * 16e-9,
                    cov30[2, 6] * 16e-9 * scf_ch4,  # kg/m2/s
                )
                * np.nanmean(1.0 * cf)
                * 1e6
            )
            # CH4 flux calculated using full corrections using point-by-point
            # corrected CH4 mass density.
            F_CH4_c = (
                ECTools.WPL_gas(
                    E,
                    air_prop["rho_air"],
                    gas_den_avg["H2O"] * 0.018,
                    S_corrected,
                    air_prop["T_avg"],
                    gas_den_avg["CH4"] * 16e-9,
                    cov30[2, 6] * 16e-9 * scf_ch4,
                )
                * 1e6
            )
            # CH4 flux calculated with temperature-related correction only using
            # point-by-point corrected CH4 mass density.
            F_CH4_T = (
                ECTools.WPL_gas(
                    0,
                    air_prop["rho_air"],
                    gas_den_avg["H2O"] * 0.018,
                    S,
                    air_prop["T_avg"],
                    gas_den_avg["CH4"] * 16e-9,
                    cov30[2, 6] * 16e-9 * scf_ch4,
                )
                * 1e6
            )
            # CH4 flux calculated using raw CH4 density with Burba et al. method
            cov_wk = cov30[2, 9]
            cov_wkc = np.nanmean(
                (data_10hz["w_rot"] - np.nanmean(data_10hz["w_rot"]))
                * (1 * cf - np.nanmean(1 * cf))
                * (data_10hz["Xmr_CH4"] - np.nanmean(data_10hz["Xmr_CH4"]))
            )
            Raw_CH4_b = (
                ECTools.WPL_gas_burba(
                    E,
                    air_prop["rho_air"],
                    gas_den_avg["H2O"] * 0.018,
                    S_corrected,
                    air_prop["T_avg"],
                    gas_den_avg["raw_CH4"] * 16e-9,
                    cov30[2, 10] * 16e-9 * scf_ch4,
                    np.nanmean(data_10hz["cf"]),
                    cov_wk * scf_ch4,
                    cov_wkc * scf_ch4 * 1e-6,
                )
                * 1e6
            )
            # CH4 flux calculated using raw CH4 density with Pan et al. method
            Raw_CH4_Pan = (
                ECTools.WPL_gas(
                    E * Mv,
                    air_prop["rho_air"],
                    gas_den_avg["H2O"] * 0.018,
                    S_corrected * MT,
                    air_prop["T_avg"],
                    gas_den_avg["raw_CH4"] * 16e-9,
                    cov30[2, 10] * 16e-9 * scf_ch4,
                )
                * 1e6
            ) * np.nanmean(data_10hz["cf"])

            H = S_corrected  # W/m2
            LE = (
                (1 + np.nanmean(data_10hz["x_H2O"]))
                * (
                    E
                    + molec_weight["H2O"]
                    * np.nanmean(data_10hz["H2O"])
                    / gas_den_avg["T"]
                    * S_corrected
                    / air_prop["rho_air"]
                    / C_p
                )
                * L_v
            )  # w/m2
            # =====================================================================
            # Calculate footprint
            # =====================================================================
            x_70 = ECTools.footprint_kljun04(cov30[2, 2], u_star, z, d, rough_len)

            # =====================================================================
            # random errors propagated through corrections
            # =====================================================================

            propagated_error_FS, raw_err_FS = ECTools.calculate_flux_error(
                err_rawCH4_stoch_FS,
                err_T_stoch_FS,
                err_h2o_stoch_FS,
                gas_den_avg,
                data_10hz,
                cov30,
                Mv,
                air_prop,
                MT,
                cov30[2, 10],
                scf_ch4=scf_ch4,
                scf_h2o=scf_h2o,
                scf_H=scf_csat3,
            )
            propagated_error_xmr_FS, raw_err_xmr_FS = ECTools.calculate_mix_flux_error(
                air_prop,
                err_xmr_stoch_FS,
                cov30,
                MT_x,
                gas_den_avg,
                err_T_stoch_FS,
                scf_ch4=scf_ch4,
                scf_H=scf_csat3,
            )
            # =====================================================================
            # Append results
            # =====================================================================
            results = pd.concat(
                [
                    results,
                    pd.DataFrame(
                        {
                            "T": air_prop["T_avg"],
                            "rho_air": air_prop["rho_air"],
                            "e": air_prop["e"],
                            "theta_v": air_prop["theta_v_avg"],
                            "p": air_prop["p_avg"],
                            "WS": WS,
                            "WD": WD,
                            "e": air_prop["e"],
                            "es": air_prop["es"],
                            "CH4": gas_den_avg["CH4"],
                            "x_CH4": np.nanmean(data_10hz["x_CH4"]),
                            "F_CH4_x_p_raw": F_CH4_x_p_raw,
                            "F_CH4_w": F_CH4_w,
                            "F_CH4_c": F_CH4_c,
                            "F_CH4_T": F_CH4_T,
                            "raw_covariance": cov30[2, 10] * 16e-3,
                            "Raw_CH4_b": Raw_CH4_b,
                            "Raw_CH4_Pan": Raw_CH4_Pan,
                            "mean_cf": np.nanmean(cf),
                            "cov_wk": cov_wk,
                            "cov_wkc": cov_wkc,
                            "CO2": gas_den_avg["CO2"],
                            "x_CO2": np.nanmean(data_10hz["x_CO2"]),
                            "F_CO2": F_CO2,
                            "H2O": gas_den_avg["H2O"],
                            "H": H,
                            "LE": LE,
                            "U": gas_den_avg["u_rot"],
                            "V": gas_den_avg["v_rot"],
                            "W": gas_den_avg["w_rot"],
                            "x_H2O": np.nanmean(data_10hz["x_H2O"]),
                            "F_H2O": E,
                            "raw_CH4": gas_den_avg["raw_CH4"],
                            "cov_uu": cov30[0, 0],
                            "cov_ww": cov30[2, 2],
                            "cov_tt": cov30[3, 3],
                            "cov_CH4CH4": cov30[6, 6],
                            "cov_h2oh2o": cov30[4, 4],
                            "cov_wt": cov30[2, 3],
                            "cov_wh2o": cov30[2, 4],
                            "cov_wco2": cov30[2, 5],
                            "cov_wCH4": cov30[2, 6] * 16e-3,
                            "cov_wrawCH4": cov30[2, 10] * 16e-3,
                            "cov_wxmr": cov30[2, 8],  # ug/g
                            "u*": u_star,
                            "L": L_ob,
                            "wpl_h2o": E - E0,
                            "wpl_co2": F_CO2 - cov30[2, 5] * 0.044,
                            "wpl_CH4": F_CH4_c - cov30[2, 6] * 0.016,
                            "spc_CH4": (cov30[2, 6] - cov30[2, 10]) * 0.016,
                            "itc_CH4": itc_CH4,
                            "itc_t": itc_t,
                            "itc_w": itc_w,
                            "itc_u": itc_u,
                            "itc_h2o": itc_h2o,
                            "itc_co2": itc_co2,
                            "sst_wt": sst["w_rotT"],
                            "sst_wCH4": sst["w_rotCH4"],
                            "sst_wh2o": sst["w_rotH2O"],
                            "sst_wco2": sst["w_rotCO2"],
                            "err_wCH4_instr": err_CH4_instr,
                            "err_wxmr_instr": err_xmr_instr,
                            "err_wh2o_instr": err_h2o_instr,
                            "err_wco2_instr": err_co2_instr,
                            "err_rawCH4_instr": err_rawCH4_instr,
                            "err_wCH4_stoch_FS": err_CH4_stoch_FS,  # umol
                            "err_wxmr_stoch_FS": err_xmr_stoch_FS,
                            "err_wh2o_stoch_FS": err_h2o_stoch_FS,
                            "err_wco2_stoch_FS": err_co2_stoch_FS,
                            "err_rawCH4_stoch_FS": err_rawCH4_stoch_FS,
                            "err_wt_stoch": err_T_stoch_FS,
                            "S": S,
                            "S_corrected": S_corrected,
                            "x_70%": x_70["x70"],
                            "na_t": na_t,
                            "na_w": na_w,
                            "na_co2": na_co2,
                            "na_CH4": na_CH4,
                            "na_w_despike": na_w_despike,
                            "na_t_despike": na_t_despike,
                            "na_co2_despike": na_co2_despike,
                            "na_h2o_despike": na_h2o_despike,
                            "na_CH4_despike": na_CH4_despike,
                            "F_CH4_x_p_coed": F_CH4_x_p_coed,
                            "F_CH4_x_c": F_CH4_x_c,
                            "MT": MT,
                            "MT_x": MT_x,
                            "Mv": Mv,
                            "coefficient": np.nanmean(coefficient - 1),
                            "water_vapor": alpha_v,
                            "instrument_error_xmr": err_xmr_instr
                            * cov30[2, 8]
                            * air_prop["rho_air"]
                            * 3600,
                            "raw_error_xmr_FS": raw_err_xmr_FS,
                            "propagated_error_xmr_FS": propagated_error_xmr_FS,
                            "instrument_error_traditional": err_rawCH4_instr
                            * cov30[2, 10]
                            * 16e-3
                            * 3600,
                            # umol/m2s * ug/umol *1e-3mg/ug *3600s/h, ->mg/m2h
                            "raw_error_traditional_FS": raw_err_FS,
                            "propagated_error_traditional_FS": propagated_error_FS,
                            "scf_h2o": scf_h2o,
                            "scf_ch4": scf_ch4,
                            "scf_csat3": scf_csat3,
                            "scf_7500": scf_7500,
                            "H2O_slow": gas_den_avg["H2O_slow"],
                            "H2O_density_8600": np.nanmean(
                                data_10hz["H2O_density_8600"]
                            ),
                            "Mv_x": Mv_x,
                            "cov_wsh2o_slow": cov30[2, 13] * 0.018,
                        },
                        index=[TS],
                    ),
                ]
            )
# %%
results.to_csv("HT8600_ECTools_results_corrected_8600_h2o_8600.csv")
