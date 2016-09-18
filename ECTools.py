"""
Notes
-----
This module includes functions for eddy covariance analysis. Many of the functions
can be used for data analysis with commercial instruments, e.g. CSAT3, LI7500, and
LI7700. Some of the functions were specifically made for QCL sensors from Zondlo's
group. Some of the functions were translated from "Eddycalc", a MATLAB package for
eddy covariance calculation.

Author
------
Da Pan,
Department of Civil and Environmental Engineering,
Princeton University
Email: dp7@princeton.edu


Created Date
------------
07/06/2016

Edited Dates
------------
07/19/2016:
    Added pandas package for time series analysis. One function "read_data_pd" was
    added. To accommodate old data structure (dictionary), pandas SettingWithCopy was
    turned off. The potential issue associated with this warning is that data may
    not be updated if the new values have different dtype, which should always be
    avoided anyway.

07/20/2016:
    Added time realignment ("time_realign") using pandas combine_first function.
    
08/20/2016:
    Added diurnaly composite analysis for long-period data plotting.  


"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.ndimage
import cPickle as pickle
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

# Define constant:

R_w = 461.5  # Ideal gas constant for water vapor, J/kg*K
R_d = 287.05  # Ideal gas constant for dry air, J/kg*K
R = 8.1  # Universal ideal gas constant, J/mol*K
L_v = 1000 * 2257  # Latent heat of vaporization (H2O), J/kg
C_p = 1005  # Approximate constant pressure specific heat of air, J/kg*K
k = 0.4  # Von Karman constant for EC calculation
g = 9.8  # Acceleration of gravity, m/s^2
gamma_d = 1.4  # Dry air specific heat ratio
alpha_v = 2.75  # NH3 foreign gas broadening coefficient for H2O density -1

a = 17.271  # Dew point coefficient a
b = 237.7  # Dew point coefficient b
sonic_angle = 265  # Sonic heading angle
sonic_num = 1  # Number of sonic;
molec_weight = {'NH3': 0.017, 'H2O': 0.018, 'CO2': 0.044, 'N2O': 0.044}

pd.set_option('chained_assignment', None)


def read_config(config_name):
    """
    Read configuration file and return configuration. Configuration should specify directory of data files (dir),
    and periods for analysis (t_start and t_end), base time specified in the LabView program (LV_time_base),
    and column names (col_names).

    Parameters
    ----------
    config_name: str
        Name of configuration file.

    Returns
    -------
    dict
        Configuration dict.

    """

    print "\nReading configuration from " + config_name + ":\n"

    config = {}

    with open(config_name) as f:
        for line in f:
            key, value = line.split(': ')
            config[key] = value[:-1]
            print key + ': ' + value[:-1]

    config['col_names'] = config['col_names'].split(', ')
    # Convert str into num for despike_lim.
    config['despike_lim'] = map(float, config['despike_lim'].split(', '))
    config['sonic_time_offset'] = float(config['sonic_time_offset'])
    return config


def save_config(config, config_name):
    """
    Save configuration into configuration file.
    Parameters
    ----------
    config: dict
        A dict containing configuration.
    config_name: str
        File name of configuration.

    Returns
    -------

    """
    with open(config_name, 'w') as f:
        for key in config.keys():
            if key in ['col_names', 'despike_lim']:
                f.write(key + ': ' + ', '.join(map(str, config[key])) + '\n')
            else:
                f.write(key + ': ' + str(config[key]) + '\n')


def get_files(dir, prefix, tInterval, date_fmt='(\d+-\d+-\d+-T\d+)\.csv'):
    """
     Get files from defined locations; File name format: prefix + '_yyyy-mm-dd-THHMM.txt'

    Parameters
    ----------
    dir: str
        Directory of raw files.
    prefix: str
        Prefix of filename
    tInterval: array of timedate64
        Time interval to read.

    Returns
    -------
    tuple (list, array)
        First element is a list of strings containing file names for later import.
        Second element is an array consisting the created datetimes of the files.


    """

    fNames = os.listdir(os.path.abspath(dir))
    outFName = []  # output file names
    dateTime = []
    for file in fNames:
        m = re.match(prefix + date_fmt,
                     file)  # match file name pattern
        if m:
            try:
                datetimestr = m.group(1)[:-2] + ':' + m.group(1)[-2:]
                datetimestr = datetimestr.replace('-T', 'T')  # Modify datatimestr

                tmp_dateTime = pd.to_datetime(datetimestr)
                print tInterval
                print tmp_dateTime
                if (tInterval[0] < tmp_dateTime) & (tInterval[1] > tmp_dateTime):
                    outFName.append(os.path.abspath(dir) + '\\' + file)
                    dateTime.append(tmp_dateTime)
            except:
                raise Exception('Wrong format, file:' + file)

    if outFName:
        return outFName, dateTime
    else:
        raise Exception('No file in the folder matches the format.')


def read_data_pd(f_names, col_names, base_time, delimiter=',', skiprows=5, \
                 time_col='Time', time_zone=-5):
    """
    Reads raw data from Zondlo's group QCL sensor. This function uses pandas package.

    Parameters
    ----------
    f_names: list of str
        Names of files to be read.
    col_names: list of str
        Names of variables.
    base_time: numpy datetime64
        Base time used in the LabView program.
    delimiter: str, default=','
        Delimiter used in the files, default=','.
    skiprows: int, default=5
        Number of header lines to be skipped.
    time_col: str, default='Time'
        Column name of time.

    Returns
    -------
    df: pandas DataFrame object
        Output data frame.

    """
    # Initiate Data
    data = pd.DataFrame()

    # Read files and append DataFrame together
    for f_name in f_names:
        tmp_df = pd.read_csv(f_name, skiprows=skiprows, names=col_names,
                             delimiter=delimiter)

        data = pd.concat([data, tmp_df], axis=0)

    data[time_col] = pd.to_timedelta(data[time_col], unit='s') + \
                     pd.to_datetime(base_time) + pd.to_timedelta(time_zone,
                                                               'h')

    data.set_index(time_col, drop=False, inplace=True)

    return data


def readData(f_names, col_names, delimiter=','):
    """
    Read data and assign the data to a dict with keys from colNames

    Parameters
    ----------
    f_names: list of str
        Names of files to read.
    col_names: list of str
        Names of columns.

    Returns
    -------
    dict
        A dictionary with keys of col_names and values of imported data.

    """

    data = {}

    for col in col_names:
        data[col] = np.array([])

    for file in f_names:
        try:
            M = np.genfromtxt(file, skip_header=6, delimiter=delimiter)
        except:
            raise Exception('Cannot read ' + file)
        if len(data.keys()) == M.shape[1]:
            for idx, col in enumerate(col_names):
                data[col] = np.hstack((data[col], M[:, idx]))
        else:
            raise Exception('Cannot read ' + file)

    return data


def mixRatio2molDen(c, p, T):
    """
    Convert mixing ratio to molar density.

    Parameters
    ----------
    c: float
        Volume mixing ratio.
    p: float
        Pressure in atm.
    T: float
        Temperature in K.

    Returns
    -------
    m: float
        Mol density in mol/m^3.

    """
    m = p * 1e3 / R / T * c
    return m


def cov_maximization(data, col_names, freq=10, lag_lim=2, idx_range=[0, -1]):
    """
    Compensate for time lag by maximizing covariance.

    Parameters
    ----------
    data: dict
        Input data.
    col_names: list of str
        Column names.
    freq: int
        Measurement frequency.
    lag_lim: int
        Limit of lag time.
    idx_range: list of int
        Start and end points of data slice.

    Returns
    -------
    tuple (array, array)
        First array contains covariances with different lag times.
        Second array contains the corresponding lag times.

    """

    # Get data for covariance calculation
    tmp_x = data[col_names[0]][idx_range[0]:idx_range[1]]
    tmp_y = data[col_names[1]][idx_range[0]:idx_range[1]]
    x = tmp_x[(~np.isnan(tmp_x)) & (~np.isnan(tmp_y))]
    y = tmp_y[(~np.isnan(tmp_x)) & (~np.isnan(tmp_y))]

    max_shift_num = int(lag_lim * freq)  # Convert lag lim to data shift limit

    cov = np.zeros((max_shift_num * 2 + 1, 1))  # Initialize output cov

    lag = np.arange(-max_shift_num, max_shift_num + 1, 1)  # Assign lag time

    # Loop to calculate covariance with varying lag times. Each loop calculates one
    # shifting x ahead in time and one shifting x behind.
    for i in range(1, max_shift_num + 1):
        # Shift backward
        m = np.c_[x[i:], y[:-i]]
        cov[max_shift_num - i] = np.abs(np.cov(np.transpose(m))[1, 0])

        # Shift forward
        m = np.c_[x[:-i], y[i:]]
        cov[max_shift_num + i] = np.abs(np.cov(np.transpose(m))[1, 0])

    m = np.c_[x, y]
    cov[max_shift_num] = np.abs(np.cov(m.T)[1, 0])

    return cov, lag


def time_realign(data, time_span='100ms'):
    """
    Realign time according to frequency.
    Parameters
    ----------
    data: DataFrame
        Input data frame.
    time_span: str, default='100ms'
        Resample time span.

    Returns
    -------
    data_out: DataFrame
        Output data frame.

    Examples
    --------
    data = pd.DataFrame({'Time': np.array(['2016-01-01T00:00:00.001234',
                                           '2016-01-01T00:00:00.101233',
                                           '2016-01-01T00:00:00.201232'],
                                          dtype='datetime64[ns]'),
                         'u': [1.5, 2.5, 3.5],
                         'v': [2.5, 3.5, 4.5]})
    data = data.set_index('Time', drop=False)
    time_realign(data)
    # Output
    # u    v                    Time
    # Time
    # 2016-01-01 05:00:00.000  1.5  2.5 2016-01-01 05:00:00.000
    # 2016-01-01 05:00:00.100  2.5  3.5 2016-01-01 05:00:00.100
    # 2016-01-01 05:00:00.200  3.5  4.5 2016-01-01 05:00:00.200

    """
    # Resample the time series using time_span.
    data_out = data.resample(time_span).mean()

    # After resample, 'Time' col will be removed. Added back here.
    data_out['Time'] = data_out.index.values

    return data_out


def molar_den2mixing(rho_gas, p, T, e):
    """
    Convert gas density to gas mixing ratio. Input could be single value or array.

    Parameters
    ----------
    rho_gas: array of float
        Air density.
    p: array of float
        Ambient pressure.
    T: array of float
        Ambieint temperature
    e: float


    Returns
    -------
    array of float
        Mixing ratio of the gas.

    """
    # p_gas = rho_gas * R_gas * T, mixing_ratio = p_gas / p
    return rho_gas * R * T / (p - e) / 1000


def get_planar_fit_coeff(u, v, w):
    """
    Calculates the planar fit coefficients for coordinate rotation (Wilczak et al., 2001).

    Parameters
    ----------
    u: array of float
        Run-averaged u (m/s)
    v: array of float
        Run-averaged v (m/s)
    w: array of float
        Run-averaged u (m/s)

    Returns
    -------
    tuple (array, array)
        (k, b)
        k: array of unit vector with three elements.
        b: tilt coefficients b0, b1, b2
    """

    # Get length of the data set
    l = len(u)
    # Calculate b using Wilczak routine
    su = np.sum(u)
    sv = np.sum(v)
    sw = np.sum(w)
    suv = np.sum(u * v)
    suw = np.sum(u * w)
    svw = np.sum(v * w)
    su2 = np.sum(u ** 2)
    sv2 = np.sum(v ** 2)
    H = np.array([[l, su, sv], [su, su2, suv], [sv, suv, sv2]])
    g = np.array([sw, suw, svw])
    b = np.linalg.lstsq(H, g)[0]

    # Calculate unit vector parallel to the new z-axis
    k = np.zeros((3,))
    k[2] = 1 / np.sqrt(1 + b[1] ** 2 + b[2] ** 2)
    k[0] = -b[1] * k[2]
    k[1] = -b[2] * k[2]

    return k, b


def dewpoint_approximation(T, RH):
    """
    Approximate dew point using Magnus formula.

    Parameters
    ----------
    T: array of float
        Air temperature.
    RH: array of float
        Relative humidity.

    Returns
    -------
    list
        T_dp: Dew point.
    """

    gamma = (a * T / (b + T)) + np.log(RH / 100.0)

    T_dp = (b * gamma) / (a - gamma)

    return T_dp


def rotate_wind_vector(u_unrot, v_unrot, w_unrot, method, k):
    """
    Transform wind to mean streamline coordinate system using double rotation, triple rotation or planar fit method.

    Parameters
    ----------
    u_unrot: array of float
        Original u (m/s)
    v_unrot: array of float
        Original u (m/s)
    w_unrot: array of float
        Original u (m/s)
    method: str
        Rotation method, i.e. 'PF': planar fit; 'DR': double rotation; 'TR':
        Triple rotation.
    k: array of float
        Unit vector of new z-axis from planar fit.

    Returns
    -------
    u_rot: array
        Rotated u (m/s).
    v_rot: array
        Rotated v (m/s).
    w_rot: array
        Rotated w (m/s).
    theta: float
        Horizontal wind direction (rad).
    phi: float
        Vertical wind tilt (rad).

    """

    # Combine winds into matrix
    wind_unrot = np.c_[u_unrot, v_unrot, w_unrot]

    # Planar fit method
    if method == 'PF':
        # Determine unit vectors i and j parallel to the new x and y axis
        j = np.cross(k, np.nanmean(wind_unrot, axis=0))
        j = j / np.sqrt(np.sum(j ** 2))
        i = np.cross(j, k)

        # Transform winds
        wind_rot = np.dot(wind_unrot, np.c_[i, j, k])
        phi = np.arccos(np.dot(k, np.array([0, 0, 1])))
        theta = np.arctan2(np.nanmean(-v_unrot), np.nanmean(u_unrot))

    else:
        # Mirror y-axes to get right-handed coordinate system (depends on the sonic)
        wind_unrot[:, 1] = -wind_unrot[:, 1]

        # First rotation to set mean(v) = 0
        theta = np.arctan2(np.nanmean(wind_unrot[:, 1]),
                           np.nanmean(wind_unrot[:, 0]))

        rot1 = np.array(
            [[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.],
             [0, 0, 1]])
        wind1 = np.dot(wind_unrot, rot1)
        # Second rotation to set mean(w) = 0
        phi = np.arctan2(np.nanmean(wind1[:, 2]), np.nanmean(wind1[:, 0]))
        rot2 = np.array([[np.cos(phi), 0, -np.sin(phi)], [0, 1, 0],
                         [np.sin(phi), 0, np.cos(phi)]])
        wind_rot = np.dot(wind1, rot2)

        # Third rotation to set mean(vw) = 0
        if method == 'TR':
            psi = 0.5 * np.arctan2(2 * np.nanmean(wind_rot[:, 1] * wind_rot[:, 2]),
                                   np.nanmean(wind_rot[:, 1] ** 2) - np.nanmean(
                                       wind_rot[:, 2] ** 2))
            rot3 = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(psi)],
                             [0, np.sin(psi), np.cos(psi)]])
            wind_rot = np.dot(wind_rot, rot3)

    u_rot = wind_rot[:, 0]
    v_rot = wind_rot[:, 1]
    w_rot = wind_rot[:, 2]

    return u_rot, v_rot, w_rot, theta, phi


def calc_sonic_T(c):
    """
    Convert sonic sound speed to air temperature.

    Parameters
    ----------
    c: float
        Sound speed.

    Returns
    -------
    float:
        Sonic temperature.

    """
    return c ** 2 / gamma_d / R_d


def calc_air_properties(Ts, H2OD, p):
    """
    Calculates average air temperature (T_avg), pressure, density (rho_air), water content (q_avg), pressure(p_avg)
    virtual potential (theta_v_avg), water vapor pressure (e), saturated water vapor pressure (es), and relative
    humidity (RH). And also return turbulent temperature, virtual potential, and water content.

    Parameters
    ----------
    Ts: array of float
        Sonic temperature.
    H2OD: array of float
        Water molar density
    p: array of float
        Ambient pressure.

    Returns
    -------
    tuple (dict, array, array, array)
        (air_prop, theta_v, q, T)
        air_prop
            Air properties, keys: T_avg, rho_air, q_avg, p_avg, e, es, RH,
            theta_v_avg
        theta_v
            Turbulent virtual potential temperature (K)
        q
            Turbulent water content (kg/m^3)
        T
            Turbulent air temperature (K)
        e
            Water vapor pressure (kPa)

    """

    # Calculate water vapor content, H2OD unit: mmol/m^3
    q = H2OD * molec_weight['H2O']

    # Calculate average pressure and temperature
    T_avg = np.nanmean(Ts)
    p_avg = np.nanmean(p)
    q_avg = np.nanmean(q)

    # Iterate to adjust water vapor effect on sonic temperature
    sonic_err = True  # Flag for if the tolerance is reached before looping for 1000 times
    i = 0  # Number of iteration
    while sonic_err & (i < 1000):
        # Calculate air temperature
        rho_air = p_avg * 1e3 / R_d / T_avg - 0.61 * q_avg
        # Adjust for
        T = Ts / (1. + 0.51 * q / rho_air)
        tmp_T_avg = np.nanmean(T)
        if np.abs(tmp_T_avg - T_avg) < 1e-3:
            sonic_err = False
        else:
            T_avg = tmp_T_avg
            i += 1

    air_prop = {}

    air_prop['T_avg'] = T_avg
    air_prop['rho_air'] = rho_air
    air_prop['q_avg'] = q_avg
    air_prop['p_avg'] = p_avg
    air_prop['e'] = q_avg * R_w * T_avg / 1000
    air_prop['es'] = 0.611 * 10. ** (7.5 * (T_avg - 273.15) / (T_avg - 273.15 +
                                                               237.3))
    air_prop['RH'] = air_prop['e'] / air_prop['es'] * 100.0

    # Calculate potential temperature
    e = q * R_w * T / 1000
    x_v = molar_den2mixing(H2OD, p, T, e)
    theta = T * (100. / p) ** (2. / 7.)
    theta_v = theta * (1. + .61 * x_v)
    air_prop['theta_v_avg'] = np.nanmean(theta_v)

    return air_prop, theta_v, q, T, e


def cosp_analysis(data, col_pairs, Time, bin_number=100, show_fig=False,
                  f_lim=[0.0005, 5]):
    """
    Perform cospectra analysis for given column pairs and save the exp-bined cospectra to out_put_files.

    Parameters
    ----------
    data: dict
        Input data.
    col_pairs: list[list[str]]
        Paris of columns for cospectra analysis. Format: [['x0','y0'],...,
        ['x_n-1','y_n-1']]
    Time: datetime64
        Time correcponding to the cospectra.
    bin_number: int
        Number of bin.
    show_fig: bool
        If true, plot the results.
    f_lim: array of float
        Frequency limits for the bin. List should have two elements.

    Returns
    -------
    dict
        Cospectra: Keys are the names of pairs, and values are binned cospectra.
        And one addition key for frequency.

    """

    cospectra = {}
    for col_pair in col_pairs:
        # Calculate cospectra
        f, cov, cos, co = calc_cospectra(data, col_pair)

        # Bin cospectra
        f_bin, cosp_bin = bin_log_spaced(f, cos, 64, f_lim=f_lim)
        cosp_bin = cosp_bin

        # Add to dict
        cospectra[col_pair[0] + col_pair[1]] = cosp_bin

        # Plot cospectra if show_fig
        if show_fig:
            # Cospectra usually are shown in loglog plot
            plt.loglog(f_bin, f_bin * cosp_bin, label=col_pair[0] + col_pair[1])

    # Since all cospectra should have same frequency bin, add the last value of f_bin to the dict
    cospectra['freq_bin'] = f_bin
    cospectra['Time'] = Time

    if show_fig:
        plt.gca().invert_xaxis()
        plt.legend()
        plt.show()

    return cospectra


def nancov(m):
    """
    Calculate covariance while ignore nan values.
    Parameters
    ----------
    m: ndarray
        Matrix for covariance calculation.

    Returns
    -------
    Cov: ndarray
        Covariance of the matrix.
    """
    valid_mask = ~np.zeros(m[0].shape, dtype='bool_')
    # This is to make sure the order agrees with original np.cov
    # Get not nan values out


    for x in m:
        valid_mask = valid_mask & ~np.isnan(x)

    m = m[:, valid_mask]
    cov = np.cov(m)

    return cov


def calc_turbulent_flux(data, air_prop, gas_avg, add_gas=None):
    """
    Calculate common turbulent fluxes of tau (surface stress, rho u'w'), H (sensible
    heat, rho cp w'T'), LE (latent heat, rho L w'x_H2O'), F_CO2 (CO2 flux,
    rho w'x_CO2'). If add_gas_flux is given, additional flux will be calculated
    using rho w'x_gas'.
    Parameters
    ----------
    data: dict
        Input data.
    rho: float
        Average air density kg/m^3.
    add_gas: list of str
        If add_flux is not none, it will be looped to calculate additional gas
        fluxes.

    Returns
    -------
    tur_flux: dict
        Keys: tau, H, LE, F_CO2, FO_H2O, w_WPL, F_gas, u_star, TKE, L
    """
    tur_flux = {}
    rho = air_prop['rho_air']
    T_avg = air_prop['T_avg']

    # Surface stress: tau = rho * u' * w'
    tur_flux['tau'] = rho * nancov(np.c_[data['u'], data['w']].T)[0, 1]

    # Sensible heat: H = rho * C_p * w' * T'
    tur_flux['H'] = rho * C_p * nancov(np.c_[data['w'], data['T']].T)[0, 1]

    # Water vapor flux: F_H2O = w' * c_H2O'
    tur_flux['F_H2O'] = nancov(np.c_[data['w'], data['H2O']].T)[0, 1]

    # Latent heat: LE = rho * L_v * w' * x_H2O'
    tur_flux['LE'] = molec_weight['H2O'] * (1 + np.nanmean(data['x_H2O'])) * (
        tur_flux['F_H2O'] + np.nanmean(data['H2O']) / T_avg * tur_flux['H'] / rho
        / C_p) * L_v

    # Calculate WPL wind velocity.
    w_WPL = WPL_velocity(rho, T_avg, tur_flux['LE'], tur_flux['H'])

    # CO2 flux: F_CO2 = rho * w' * x_CO2
    tur_flux['F_CO2'] = molec_weight['CO2'] * ((np.c_[data['w'], data['CO2']].T)[
                                                   0, 1] + gas_avg['CO2'] * w_WPL)

    tur_flux['w_WPL'] = w_WPL

    # Flux of additional gases
    for gas in add_gas:
        tur_flux['F_' + gas] = molec_weight[gas] * \
                               (nancov(np.c_[data['w'], data[gas]].T)[0, 1] + \
                                gas_avg[gas] * w_WPL)

    # Turbulence properties
    wind_cov = nancov(np.c_[data['u'], data['v'], data['w']].T)

    # u_star = ((u'w')^2 + (v'w')^2)^1/4
    tur_flux['u_star'] = (wind_cov[0, 2] ** 2 + wind_cov[1, 2] ** 2) ** 0.25

    # TKE = 1/2 * (u'^2 + v'^2 + w'^2)
    tur_flux['TKE'] = 0.5 * (wind_cov[0, 0] + wind_cov[1, 1] + wind_cov[2, 2])

    # w'theta_v'
    tmp_theta_cov = nancov(np.c_[data['w'], data['theta_v']].T)[0, 1]

    # theta_v_avg
    theta_v_avg = np.nanmean(data['theta_v'])

    tur_flux['L'] = -tur_flux['u_star'] ** 3. * theta_v_avg / k / g / tmp_theta_cov

    tur_flux['wind_direction'] = np.degrees(np.nanmean(data['theta']))

    tur_flux['wind_speed'] = np.sqrt(
        np.nanmean(data['u_rot'] ** 2 + data['v_rot'] ** 2))

    return tur_flux


def shift_lag(data, col_names, target_col, lag_lim=10, idx_range=[0, -1],
              show_fig=False, freq=10, create_fig=True, fig=None, axes=None):
    """
    This function calls cov_maximization and shift the time lag.

    Parameters
    ----------
    data: dict
        Input data.
    col_names: list of str
        Column names for shifting.
    target_col: str
        The column that other columns will shift towards to.
    lag_lim: int
        Lat time limits (second).
    idx_range: list of int
        Index range.
    show_fig: bool
        If true, show results in a figure.
    freq: int
        Measurement frequency.

    Returns
    -------

    """
    # Create handles for figure
    if show_fig:
        if create_fig:
            fig, axes = plt.subplots(len(col_names), sharex=True)

    max_shift_num = int(lag_lim * freq)

    for idx, col in enumerate(col_names):
        cov, lag = cov_maximization(data, [col, target_col], lag_lim=lag_lim,
                                    idx_range=idx_range, freq=freq)

        # Plot lag against covariance
        if show_fig:
            axes[idx].plot(lag * 0.1, cov)
            axes[idx].set_ylabel(col_names[idx])

        lag_num = np.argmax(cov) - max_shift_num
        if lag_num < 0:
            data[col][0:lag_num] = data[col][-lag_num:]
            data[col][lag_num:] = np.nan
        elif lag_num > 0:
            data[col][lag_num:] = data[col][0:-lag_num]
            data[col][0:lag_num] = np.nan

    if show_fig:
        if create_fig:
            plt.show()
        else:
            fig.tight_layout()
            fig.canvas.draw()


def calc_cospectra(data, col_names, freq=10):
    """
    Calculate normalized cospectra density. Detrend and Bell taper should always be performed before running this
    function. For normal spectra calculation, use the same col_names.

    Parameters
    ----------
    data: dict
        Input data.
    col_names: list of str
        Column names for cospectra calculation.
    freq: int
        Measurement frequency (Hz), default = 10 Hz
    Returns
    -------
    tuple (array, float, array)
        (f, cov, cospectra)
        f
            frequency for the cospectra.
        cov
            covariance of the pair.
        cospectra
            Normalized cospectra density.
    """

    # Remove nan data
    tmp_x = data[col_names[0]]
    tmp_y = data[col_names[1]]
    x = tmp_x[(~np.isnan(tmp_x)) & (~np.isnan(tmp_y))]
    y = tmp_y[(~np.isnan(tmp_x)) & (~np.isnan(tmp_y))]

    l = len(x)

    if l != len(y):
        raise Exception('Length of data records must agree')
    if l > 1:
        # Bell taper the input data
        x = bell_taper(x)
        y = bell_taper(y)

        # Calculate frequency bin according to the length of input data
        if l % 2 == 0:
            f = freq * np.linspace(0, 1, l + 1)[0:-2]
        else:
            f = freq * np.linspace(0, 1, l)

        df = f[1]  # Frequency interval
        f = f[1:np.floor(l / 2)]  # Frequency up to Nyquist frequency

        m = np.transpose(np.c_[x, y])
        cov = np.cov(m)[1, 0]  # Covariance

        # Calculate discrete Fourier transform
        fft_x = np.fft.fft(x) / l
        fft_y = np.fft.fft(y) / l
        f = np.fft.fftfreq(l, 1. / freq)
        Co = np.real(np.conj(fft_x) * fft_y)

        if l % 2 == 1:
            Co = 2 * Co[1:np.floor(l / 2)] / cov
            f = f[1:np.floor(l / 2)]
        else:
            Co = np.r_[
                2 * Co[1:np.floor(l / 2)] / cov, 2 * Co[
                    np.floor(l / 2) + 1] / cov]
            f = np.r_[f[1:np.floor(l / 2)], np.floor(l / 2) + 1]

        cospectra = Co / df

    else:
        f = [1, 10]
        cov = np.array([np.nan, np.nan])
        cospectra = np.array([np.nan, np.nan])
        Co = np.array([np.nan, np.nan])

    return f, cov, cospectra, Co


def shift_sonic_time(data, col_names, sonic_time_offset, freq=10):
    """
    Shift sonic time for synchronization.
    Parameters
    ----------
    data: DataFrame
        Input data containing at least 'u', 'v', 'w', and 'Ts'.
    sonic_time_offset: float
        Time in sec for shifting.
    freq: float, default=10
        Measurement frequency in Hz.

    Returns
    -------

    """
    # col_names = ['u', 'v', 'w', 'Ts']
    lag_num = int(sonic_time_offset * freq)

    for col in col_names:
        if lag_num < 0:
            data[col][0:lag_num] = data[col][-lag_num:]
            data[col][lag_num:] = np.nan
        elif lag_num > 0:
            data[col][lag_num:] = data[col][0:-lag_num]
            data[col][0:lag_num] = np.nan


def bin_log_spaced(f, cospectra, n, f_lim=None):
    """
    Bin cospectra into n log-spaced bins.

    Parameters
    ----------
    f: array of float
        Frequency vector calculated from calc_cospectra.
    cospectra
        Cospectra vector calculated from calc_cospectra.
    n: int
        Number of bins.
    f_lim
        Frequency limits for the bin. List should have two elements.
    Returns
    -------
    f_out: array of float
        Binned frequency.
    cosp_out: array of float
        Binned cospectra.
    """

    if f_lim is None:
        min_f = np.min(f)
        max_f = np.max(f)
    else:
        min_f = f_lim[0]
        max_f = f_lim[1]

    bounds = np.zeros((n + 1,))
    bounds[0] = 0
    bounds[1:n] = np.exp(np.linspace(np.log(min_f), np.log(max_f), n - 1))

    f_out = np.zeros((n,))
    cosp_out = np.zeros((n,))

    for i in range(0, n):
        bin = (f > bounds[i]) & (f < bounds[i + 1])
        f_out[i] = np.exp((np.log(bounds[i]) + np.log(bounds[i + 1])) / 2)
        cosp_out[i] = np.nanmean(cospectra[bin])

    return f_out, cosp_out


def avg_one_min_data(data, col_names, out_f_name, show_fig=False):
    """
    Calculate one-minute averaged data.

    Parameters
    ----------
    data: dict
        Input data.
    col_names: list of str
        Names of columns for averaging.
    out_f_name: str
        Output file name.
    show_fig: bool
        If True, show figure. Default False.
    Returns
    -------

    """

    one_min_data = chunk_data(data, t_window=1)
    one_min_avg = {'Time': []}

    # Create array for each
    for col in col_names:
        one_min_avg[col] = []

    for data in one_min_data:

        one_min_avg['Time'].append(data['Time'][300])  # Get the mid time

        for col in col_names:
            one_min_avg[col].append(np.nanmean(data[col]))  # Add new avg value

    one_min_avg['Time'] = np.array(one_min_avg['Time'], dtype='datetime64')
    for col in col_names:
        one_min_avg[col] = np.array(one_min_avg[col])

    if show_fig:
        plot_time_series(one_min_avg, col_names, idx_range=[0, -1])

    pickle.dump(one_min_avg, open(out_f_name, 'w'))


def bell_taper(x_in, alpha=0.1):
    """
    Bell tapering for the data block. A process of modifying a time series of data whereby the beginning and ending 10%
    of the data points are smoothly reduced in amplitude to approach zero at the ends.
    For more info: http://glossary.ametsoc.org/wiki/Bell_taper

    Parameters
    ----------
    x_in: array of float
        Time series for bell tapering, here an array should be passed in.
    alpha: float
        Proportion of the series to be tapered at each end of the series (default 0.1)
    Returns
    -------
    x_out: array of float
        Output results

    """

    l = len(x_in)
    m = np.floor(alpha * l)

    taper_func = 0.5 * (1 - np.cos(np.pi * (np.arange(0, m) + 0.5) / m))

    x_out = np.copy(x_in)
    x_out[0:m] = taper_func * x_out[0:m]
    x_out[-m:] = taper_func[::-1] * x_out[-m:]

    return x_out


def detrend(data, col_names, method='linear'):
    """
    Remove trend from input data. Nan data will be considered as missing data.
    Parameters
    ----------
    data: dict
        Data for detrending.
    col_names: list of str
        Column names for detrending.
    method: str
        Detrending method, default is "linear"
    Returns
    -------

    """
    data_out = data.copy()

    if method == 'linear':
        for col in col_names:
            # print data_out[col]
            idx = np.where(~np.isnan(data_out[col]))[0]
            if len(idx) == 0:
                print 'Empty period.'
                data_out[col] = data_out[col]
            else:
                y = data_out[col][idx]
                # print idx, y
                p = np.polyfit(idx, y, 1)
                data_out[col] = data_out[col] - p[1] - \
                                p[0] * np.arange(0, len(data_out[col]))

    return data_out


def despike(data, col_names, lim, n=10, method='constant', diag=False):
    """
    Despike input data using median filter. If the value at given time has a deviation from median values of n nearby
    data points, the value will be marked as NaN.

    Parameters
    ----------
    data: dict
        Input data.
    col_names: list of str
        Column names for despiking.
    lim: array of float
        A array lists the lim for excluding data arranged in accordance with
        col_names
    n: int
        Size of median filter, default is 10
    method: str
        Choose from 'constant' (use constant to exclude spikes) or 'std' (use
        multiple of standard deviation to exlude spikes).

    Returns
    -------

    """

    n_spikes = np.zeros((len(col_names),))

    print len(lim)
    print len(col_names)

    for idx, col in enumerate(col_names):

        idx_nonnan = ~np.isnan(data[col])

        tmpData = data[col][idx_nonnan]

        # Call scipy median filter
        data_filt = scipy.ndimage.median_filter(tmpData, size=n)

        # Calculate deviation from median filtered data
        data_dev = np.abs(tmpData - data_filt)

        # Replace spikes with nans
        if method == 'std':
            tmpData[data_dev > lim[idx] * np.nanstd(data_dev)] = np.nan
        else:
            tmpData[data_dev > lim[idx]] = np.nan

        data[col][idx_nonnan] = tmpData

        if diag:
            print np.nanstd(data[col])
            plt.plot(data_dev)
            plt.show()


def slice_data(data, idx_range=None, time_range=None):
    """
    Slice data according to index range.

    Parameters
    ----------
    data: dict
        Input data
    idx_range: list
        Index range with two elements
    time_range: list
        Time range

    Returns
    -------
    dict
        Sliced data
    """
    new_data = {}

    # Get new data using index range.
    if idx_range is not None:
        for col in data.keys():
            new_data[col] = data[col][idx_range[0]:idx_range[1]]

    # Get new data using time range.
    if time_range is not None:
        for col in data.keys():
            new_data[col] = data[col][(data['Time'] > time_range[0]) & (data['Time']
                                                                        < time_range[
                                                                            1])]
    return new_data


def chunk_data(data_in, t_window=30, start_time=None, correct_time=99.9, freq=10):
    """
    This function chunks data into pieces with according to time.

    Parameters
    ----------
    data_in: dict
        Input data.
    t_window: float
        Chunk data into blocks accordingly, unit: min.
    start_time: datetime64
        Start time of data block, if not set the first time stamp will be used.
    correct_time: float
        Factor to correct time scale difference (unit: ms)
    freq: int
        Measurement frequency
    Returns
    -------
    data_out: dict
        Data_out is a list containing all the time periods. In addition to original
        dict keys, t_start for each period is also included.
    """

    # Define the t_start for the first data block
    if start_time is not None:
        t_start = start_time
    else:
        t_start = data_in['Time'][0]

    l = t_window * freq * 60  # Length of each data block

    # Convert t_window into numpy.timedelta64
    t_window = np.timedelta64(t_window, 'm')

    data_out = []  #

    while t_start < data_in['Time'][-1]:
        block = {}  # Temperoral place to store data block

        # Get time-based indexing mask
        t_mask = (data_in['Time'] > t_start) & (data_in['Time'] < t_start + t_window)

        # For some period the real measurement frequency is 9.99 Hz, following codes correct the time
        # and put them into an array that has 18000 elements
        if correct_time is not None:
            tmp_time = data_in['Time'][t_mask]
            # ms_since_start: milliseconds since start
            ms_since_start = np.int32(np.ceil(np.float64(
                (tmp_time - t_start) / correct_time)))  # Sometimes ceil works
            if np.nansum(np.diff(ms_since_start) == 0) > 0:
                ms_since_start = np.int32(
                    np.floor(np.float64((
                                            tmp_time - t_start) / correct_time)))  # Sometimes floor works
            idx = np.where(ms_since_start < l)
            ms_since_start = ms_since_start[ms_since_start < l]  # Remove >18000
        else:
            idx = np.arange(0, l)
            ms_since_start = np.arange(0, l)

        # Assign t_start value
        block['t_start'] = t_start
        # Pass input data into block
        for key in data_in.keys():
            block[key] = np.full((l,), np.nan)
            block[key][ms_since_start] = data_in[key][t_mask][idx]

        block['Time'] = np.arange(t_start, t_start + t_window,
                                  np.timedelta64(100, 'ms'))
        # Add block to output data
        data_out.append(block)

        # Move on to next period
        t_start = t_start + t_window

    # Delete last block if the number of elements < 12000
    if np.sum(np.isnan(data_out[-1][key])) > 6000:
        data_out = data_out[:-2]

    return data_out


def calc_avg(data, col_names):
    """
    Calculate block-averaged gas concentration (mass and molar density and mixing ratio).

    Parameters
    ----------
    data: dict
        Input data.
    col_names: list of str
        Column names of gases.
    Returns
    -------
    avg_values: dict
        Averaged results.
    """

    avg_values = {}

    for col in col_names:
        avg_values[col] = (np.nanmean(data[col]))

    return avg_values


def QC_QCL_NH3(data, col_names, peak_loc, peak_thold, intensity_thold, t_window=1,
               freq=10):
    """
    Quality control for QCL NH3 sensor.

    Parameters
    ----------
    data: dict
        Input data.
    col_names: list of str
        Column name of the peak location and intensity
    peak_loc: int
        Set point of peak location.
    peak_thold: int
        Threshold for d4etermining unstable laser temperature.
    intensity_thold:
        Threshold of intensity for removing low intensity results.
    t_window: int
        Remove t_winodow before and after the unstable point, unit: min
    freq: int
        Measurement frequency.
    Returns
    -------
    invalid_mask: list of bool
        A mask that can be used to remove bad data.

    """

    # Find the indices for unstable temperature
    b_idx = np.where((np.abs(data[col_names[0]] - peak_loc) > peak_thold) | (
        data[col_names[1]] < intensity_thold))[0]

    idx_window = t_window * 60 * freq

    dlen = len(data[col_names[0]])  # Data length

    # Mask b_idx +- t_window as invalid data
    invalid_mask = np.zeros((dlen,), dtype='bool_')

    for idx in b_idx:
        idx_start = max(idx - idx_window, 0)
        idx_end = min(idx + idx_window, dlen)
        invalid_mask[idx_start:idx_end] = True

    return invalid_mask


def plot_time_series(data, col_names, time_col='Time', idx_range=None, sharex=True,
                     label=None, save_fig=False, fig_name=None):
    """
    This function provides basic function for plotting NH3 EC results in multiple
    panels.

    Parameters
    ----------
    data: dict
        Data to plot.
    col_names: list of str
        Name of variables.
    idx_range: list of int
        Index range.
    sharex: bool
        if True subplots will share x-axis.
    save_fig: bool
        If True, figure will be saved.
    Returns
    -------
    fig: object
        Figure handle.
    axes: list of objects
        Handles of axes.

    """

    n = len(col_names)  # Number of panelsdx

    # Create figure using subplots each panel has a height of 3 inches
    fig, axes = plt.subplots(n, sharex=sharex, figsize=(8, n * 3))

    # Convert type of Time data for x axis formatter
    xdate = data[time_col].astype('M8[ms]').astype('O')

    # Loop plotting each variable
    for idx, ax in enumerate(axes):
        if idx_range is None:
            if label is None:
                ax.plot(xdate, data[col_names[idx]])
                ax.set_ylabel(col_names[idx])
            else:
                ax.plot(xdate, data[col_names[idx]])
                ax.set_ylabel(label[idx])
        else:
            ax.plot(xdate[idx_range[0]:idx_range[1]],
                    data[col_names[idx]][idx_range[0]:idx_range[1]], '.')
            ax.set_ylabel(col_names[idx])

    xfmt = mdates.DateFormatter('%m-%d %H:%M')
    axes[-1].xaxis.set_major_formatter(xfmt)
    if save_fig:
        pickle.dump(fig, open(fig_name, 'w'))

    plt.tight_layout()
    plt.show()
    return fig, axes


def WPL_velocity(rho, T_avg, LE, H):
    """
    Calculate WPL velocity. Using method describe on Pg. 129 - 130, Handbook of
    Micrometeorology(Lee et al.).

    Parameters
    ----------
    rho: float
        Air density.
    T_avg: float
        Air temperature.
    LE: float
        Latent heat flux.
    H: float
        Snesible heat flux.

    Returns
    -------
    w_WPL: float
        WPL air velocity term.
    """
    return LE / rho * R_d / R_w / L_v + H / rho / C_p / T_avg


def combine_output_data(data_out, target_gas, Time, air_prop, gas_den, tur_flux, sst,
                        itc):
    """
    Combine results from different functions.

    Parameters
    ----------
    data_out: dict
        A dict to put results in. Keys: Time, T_avg, rho_avg, q_avg, p_avg, e, es, RH,
        theta_v_avg, H2O_avg, CO2_avg, NH3_avg, tau, H, LE, F_CO2, F_H2O, F_NH3,
        w_WPL, TKE, L
    air_prop: dict
        Results of air properties.
    gas_den: dict
        Averaged gas density results.
    tur_flux: dict
        Results of turbulent fluxes.
    Returns
    -------

    """
    data_out['Time'].append(Time)
    data_out['T_avg'].append(air_prop['T_avg'])
    data_out['rho_air'].append(air_prop['rho_air'])
    data_out['q_avg'].append(air_prop['q_avg'])
    data_out['p_avg'].append(air_prop['p_avg'])
    data_out['e'].append(air_prop['e'])
    data_out['es'].append(air_prop['es'])
    data_out['RH'].append(air_prop['RH'])
    data_out['theta_v_avg'].append(air_prop['theta_v_avg'])
    data_out['H2O_avg'].append(gas_den['H2O'])
    data_out['CO2_avg'].append(gas_den['CO2'])
    data_out[target_gas + '_avg'].append(gas_den[target_gas])
    data_out['tau'].append(tur_flux['tau'])
    data_out['H'].append(tur_flux['H'])
    data_out['LE'].append(tur_flux['LE'])
    data_out['F_CO2'].append(tur_flux['F_CO2'])
    data_out['F_H2O'].append(tur_flux['F_H2O'])
    data_out['F_' + target_gas].append(tur_flux['F_' + target_gas])
    data_out['w_WPL'].append(tur_flux['w_WPL'])
    data_out['TKE'].append(tur_flux['TKE'])
    data_out['L'].append(tur_flux['L'])
    data_out['SST_T'].append(sst['wT'])
    data_out['SST_H2O'].append(sst['wH2O'])
    data_out['SST_CO2'].append(sst['wCO2'])
    data_out['SST_' + target_gas].append(sst['w' + target_gas])
    data_out['ITC'].append(itc)
    data_out['u_star'].append(tur_flux['u_star'])
    data_out['wind_direction'].append(tur_flux['wind_direction'])
    data_out['wind_speed'].append(tur_flux['wind_speed'])


def read_TOA5(file, time_col='Time', time_zone=-5):
    """
    This function reads TOA files. This type of file is usually used by MSU.
    Parameters
    ----------
    files: list
        Filenames of the TOA files.

    Returns
    -------
    data: dict
        Output dict contains Ts, u, v, w, CO2, H2O, T, P, diag_csat, and agc.

    """
    data = pd.read_csv(file, skiprows=4, low_memory=False,
                       names=["Time", "RECORD", "u", "v", "w", "CO2",
                              "H2O", "Ts", "p", "diag_csat", "agc"],
                       parse_dates=True, na_values='NAN')

    data[time_col] = pd.to_datetime(data[time_col]).values + \
                     np.timedelta64(time_zone, 'h')

    data.set_index(time_col, drop=False, inplace=True)

    return data


def data_append(data_in, data_add):
    """
    This function appends two dict like data together. No returns.
    Parameters
    ----------
    data_in: dict
        Input data, and it will be appended.
    data_add: dict
        Data to be added to input data.

    Returns
    -------

    """

    for col in data_in.keys():
        data_in[col].append(data_add[col])


def create_output_data(target_gas):
    """
    Create the output dictionary.
    Returns
    -------

    """
    data_out = {'Time': [], 'T_avg': [], 'rho_air': [], 'q_avg': [], 'p_avg': [],
                'e': [], 'es': [], 'RH': [], 'theta_v_avg': [], 'H2O_avg': [],
                'CO2_avg': [], target_gas + '_avg': [], 'tau': [], 'H': [], 'LE': [],
                'F_CO2': [], 'F_H2O': [], 'F_' + target_gas: [], 'w_WPL': [],
                'TKE': [], 'L': [], 'SST_T': [], 'SST_H2O': [], 'SST_CO2': [],
                'SST_' + target_gas: [], 'ITC': [], 'u_star': [],
                'wind_direction': [], 'wind_speed': []}

    return data_out


def plot_time_series_pd(data, col_names, fig_size=(6, 4), sharex=True, labels=None,
                        time_span='Days', create_fig=True, fig=None, axes=None):
    """
    Plot time series for Pandas DataFrame.
    Parameters
    ----------
    data: DataFrame
        Input data frame.
    col_names: list of str
        Names of columns to be plotted.
    fig_size: tuple, default=(6,4)
        Figure size parameters to be passed to plt.subplots
    sharex: bool, default=True
        If True, share a-axis.
    time_span: str, default='Days'
        Time span of the time series, should be one of the following:
        'Months', 'Days', 'Hours', and 'Minutes'.

    Returns
    -------
    fig: handle
        Figure handle.
    axes: list of handles                     
        Axes handles.

    """
    
    # Create
    if create_fig:
        fig, axes = plt.subplots(len(col_names), figsize=fig_size, sharex=sharex)

    if labels is None:
        labels = col_names

    for ax, col, label in zip(axes, col_names, labels):
        ax.plot_date(data.index, data[col], '-', label=label)
        ax.set_ylabel(label)

    plt.setp(ax.get_xticklabels(), visible=True)
    # Modify date formatter according to time span.
    if time_span == 'Months':
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    if time_span == 'Weeks':
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    if time_span == 'Days':
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    if time_span == 'Hours':
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

    if time_span == 'Minutes':
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    if create_fig:
        plt.show()
    else:
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.canvas.draw()

    return fig, axes


def stationarity_test(data, col_pairs, n=5):
    """
    Calculates parameters for stationarity test. Data will be sliced into n
    sub-array, and SST will be calculated. sst = (mean(cov_sub) - cov)/cov * 100.
    Parameters
    ----------
    data: DataFrame
        Input data.
    col_pairs: list of list of str
        Names of columns for the calculation.
    n: int
        Number of sub arrays.

    Returns
    -------
    sst: dict-like
        Contains sst results for given columns.

    """
    tmp_data = data.copy()
    sst = {}

    # Get length of the data
    l = len(tmp_data[col_pairs[0][0]])

    # Use linspace to get n+1 index for slices and combine them to range
    idx = np.floor(np.linspace(0, l, num=n))
    ranges = zip(idx[0:-1], idx[1:])

    # Initiate an array to store covariances of sub slices.
    cov_sub = np.full((n,), np.nan)

    # Loop through pairs
    for pair in col_pairs:
        # Calculate covariance for the pair
        m = np.c_[tmp_data[pair[0]], tmp_data[pair[1]]].T
        cov = nancov(m)[0, 1]
        # Loop through sub slices
        for i, rng in enumerate(ranges):
            m = np.c_[tmp_data[pair[0]].values[rng[0]:rng[1]],
                      tmp_data[pair[1]].values[rng[0]:rng[1]]].T
            cov_sub[i] = nancov(m)[0, 1]

        sst[pair[0] + pair[1]] = (np.nanmean(cov_sub) - cov) / cov * 100

    return sst


def itc_test(data, obukhov_l, d, z, var_w2, u_star):
    """
    Integral turbulence characteristics (itc) test.
    Parameters
    ----------
    data: DataFrame
        Input data.
    obukhov_l: float
        Obukhov length.
    d: float
        Displacement height.
    z: float
        Height of the instrument.

    Returns
    -------
    itc: float

    """
    stability_parameter = (z - d) / obukhov_l

    if stability_parameter > 1:
        sigma_uw_theory = 2
    elif stability_parameter < 0.032:
        sigma_uw_theory = 1.3
    else:
        sigma_uw_theory = 2 * np.abs(
            stability_parameter) ** 0.125  # 0.032 <= |z/L| <= 1

    return (sigma_uw_theory - np.sqrt(var_w2)) / u_star / sigma_uw_theory * 100


def count_non_nan(x):
    """
    Count total number of non-nan values.
    Parameters
    ----------
    x: array-like

    Returns
    -------
    count: int
    """
    return np.sum(~np.isnan(x)).astype('int64')


def calc_turbulent_flux_pd(data, target_gas, d, z, alpha_v,
                           p_cal, T_cal, df_cf, t_window='30min',
                           save_30min=False, save_all=False, show_fig=True,
                           calc_Ts=False, lag_lim=20, diag=False, create_fig=True,
                           fig=None, axes=None):
    """
    Calculate turbulent fluxes using Pandas DataFrame.

    Parameters
    ----------
    data: DataFrame
        Input DataFrame, usually should contain data for one day.
    target_gas: str
        Name of target gas of QCL sensor.
    d: float
        Displacement height, unit: m.
    z: flozt
        Tower height, unit: m.
    alpha_v:
        Foreign gas broadening coefficient for H2O.
    p_cal: float
        Calibration pressure in kPa.
    T_cal:
        Calibration temperature in K.
    df_cf: DataFrame
        DataFrame that contains information for spectroscopic correction.
    t_window: str, default='30min'
        Time window for average and calculation of fluxes.
    save_30min: bool, default=False
        If true, save 30-min data to individual csv file.
    save_all: bool. default=False
        If true, save combined data into one csv file.
    show_fig: bool, default=True
        If true, show figures for diagnostics.
    calc_Ts: bool, default=False
        If true, onvert sound speed to sonic temperature.
    lag_lim: int, default=20
        Limit of lag time in second.

    Returns
    -------
    block_data_out: dict
        Dict contains tau, H, LE, F_CO2, F_H2O, F_target_gas, w_WPL, L,
        u_star, TKE, T_avg, rho_air, theta_v_avg, e, es, RH, SST, and ITC
    cospectra_out: dict
        Dict contains wCO2, wH2O, w+target_gas, wT, L, and u_star

    """
    # Despike input data.
    block_data_out = create_output_data(target_gas)
    cospectra_out = []
    if calc_Ts:
        data['Ts'] = calc_sonic_T(data['c'])

    despike(data, [target_gas, 'H2O', 'CO2', 'u', 'v', 'w', 'Ts'],
            [10, 200, 5, 3, 3, 3, 3], n=10)

    # Count valid data number for target gas
    count = data.resample(t_window).apply(count_non_nan)
    t_delta = pd.to_timedelta(t_window)

    # Get interp2d object for spectroscopic correction.
    spec_correction = LinearNDInterpolator(
        np.c_[df_cf['p_grid'], df_cf['t_grid']], df_cf['correction_factor'])

    for t_idx in count.index:
        if sum(count.loc[t_idx, [target_gas, 'CO2', 'H2O', 'w', 'Ts']] > 12000) == 5:

            tmp_data = data[t_idx: t_idx + t_delta]

            shift_lag(tmp_data, [target_gas, 'H2O', 'CO2'], 'w', lag_lim=lag_lim,
                      show_fig=show_fig, create_fig=create_fig, fig=fig, axes=axes)

            tur_flux, air_prop, gas_den_avg, cospectra, sst, \
            itc = calc_turflux_30min(
                tmp_data, t_idx,
                target_gas, d, z,
                alpha_v, p_cal,
                T_cal, spec_correction)

        else:
            tur_flux, air_prop, gas_den_avg, cospectra, sst, \
            itc = fill_null(target_gas)

        combine_output_data(block_data_out, target_gas, t_idx.to_datetime(),
                            air_prop,
                            gas_den_avg, tur_flux, sst, itc)
        cospectra_out.append(cospectra)

    if save_all:
        data.to_csv(
            target_gas + '_L2_all_' + t_idx.strftime('%Y-%m-%dT%H:%M') + '.csv')

    return block_data_out, cospectra_out


def calc_turflux_30min(tmp_data, t_idx, target_gas, d, z, alpha_v,
                       p_cal, T_cal, spec_correction, save_30min=False):
    """

    Parameters
    ----------
    data: DataFrame
        Input DataFrame, usually should contain data for one day.
    target_gas: str
        Name of target gas of QCL sensor.
    d: float
        Displacement height, unit: m.
    z: flozt
        Tower height, unit: m.
    alpha_v:
        Foreign gas broadening coefficient for H2O.
    p_cal: float
        Calibration pressure in kPa.
    T_cal:
        Calibration temperature in K.
    spec_correction: scipy.interpolate object
        A function that returns interpolated values.
    t_window: str, default='30min'
        Time window for average and calculation of fluxes.
    save_30min: bool, default=False
        If true, save 30-min data to individual csv file.
    save_all: bool. default=False
        If true, save combined data into one csv file.
    calc_Ts: bool, default=False
        If true, onvert sound speed to sonic temperature.

    Returns
    -------
    tur_flux: dict
        Turbulent flux results.
    air_prop: dict
        Air properties.
    gas_den_avg: dict
        Gas density averaged for 30 min.
    cospectra: dict
        Cospectra output.
    sst: dict
        Stationarity test results.
    itc: dict
        Integrated turbulence test results.

    """
    if save_30min:
        tmp_data.to_csv(target_gas + '_L2_30min_' + t_idx.strftime(
            '%Y-%m-%dT%H:%M') + '.csv')

    tmp_data['u_rot'], tmp_data['v_rot'], tmp_data['w_rot'], \
    tmp_data['theta'], tmp_data['phi'] = rotate_wind_vector(
        tmp_data['u'], tmp_data['v'], tmp_data['w'], 'TR', k=None)

    # Convert unit of gas to mol/m^3
    tmp_data['H2O'] = tmp_data['H2O'] * 1e-3
    tmp_data['CO2'] = tmp_data['CO2'] * 1e-3
    tmp_data[target_gas] = tmp_data[target_gas] * mixRatio2molDen(1e-9,
                                                                  p_cal,
                                                                  T_cal)

    air_prop, tmp_data['theta_v'], tmp_data['q'], tmp_data['T'], \
    tmp_data['e'] = calc_air_properties(tmp_data['Ts'], tmp_data['H2O'],
                                        tmp_data['p'])

    # Get mixing ratio of gases for WPL correction

    tmp_data['x_H2O'] = molar_den2mixing(tmp_data['H2O'], tmp_data['p'],
                                         tmp_data['T'], tmp_data['e'])
    tmp_data['x_CO2'] = molar_den2mixing(tmp_data['CO2'], tmp_data['p'],
                                         tmp_data['T'], tmp_data['e'])

    # Spectroscopic correction
    p_e = tmp_data['p'] * (1 + alpha_v * tmp_data['x_H2O']) * 10.0
    T = tmp_data['T']
    cf = spec_correction(p_e.values, T)

    tmp_data[target_gas] = tmp_data[target_gas] / cf

    tmp_data['x_' + target_gas] = molar_den2mixing(tmp_data[target_gas],
                                                   tmp_data[
                                                       'p'],
                                                   tmp_data['T'],
                                                   tmp_data['e'])

    gas_den_avg = calc_avg(tmp_data, [target_gas, 'H2O', 'CO2'])

    tmp_data = detrend(tmp_data, [target_gas, 'H2O', 'CO2', 'T', 'u_rot',
                                  'v_rot', 'w_rot'])

    cospectra = cosp_analysis(tmp_data, [['w', 'T'], ['w', 'H2O'],
                                         ['w', 'CO2'], ['w', target_gas]],
                              t_idx.to_datetime())

    tur_flux = calc_turbulent_flux(tmp_data, air_prop, gas_den_avg,
                                   add_gas=[target_gas])

    cospectra['L'] = tur_flux['L']
    cospectra['u_star'] = tur_flux['u_star']

    sst = stationarity_test(tmp_data, [['w', 'T'], ['w', 'H2O'],
                                       ['w', 'CO2'], ['w', target_gas]], 6)
    itc = itc_test(tmp_data, tur_flux['L'], d, z, np.nanvar(tmp_data['w'] ** 2.0),
                   tur_flux['u_star'])

    return tur_flux, air_prop, gas_den_avg, cospectra, sst, itc


def customized_box_plot(percentiles, fig, axes, redraw=True, solid_whisker=True,
                        same_color=None, *args, **kwargs):
    """
    Generates a customized boxplot based on the given percentile values
    """

    n_box = len(percentiles)

    box_plot = axes.boxplot([[-9, -4, 2, 4, 9], ] * n_box, *args, **kwargs)
    # Creates len(percentiles) no of box plots

    min_y, max_y = float('inf'), -float('inf')

    for box_no, (q1_start,
                 q2_start,
                 q3_start,
                 q4_start,
                 q4_end) in enumerate(percentiles):

        # Lower cap
        box_plot['caps'][2 * box_no].set_ydata([q1_start, q1_start])
        # xdata is determined by the width of the box plot

        # Lower whiskers
        box_plot['whiskers'][2 * box_no].set_ydata([q1_start, q2_start])
        if solid_whisker:
            box_plot['whiskers'][2 * box_no].set_linestyle('-')

        # Higher cap
        box_plot['caps'][2 * box_no + 1].set_ydata([q4_end, q4_end])

        # Higher whiskers
        box_plot['whiskers'][2 * box_no + 1].set_ydata([q4_start, q4_end])
        if solid_whisker:
            box_plot['whiskers'][2 * box_no + 1].set_linestyle('-')

        # Box
        box_plot['boxes'][box_no].set_ydata([q2_start,
                                             q2_start,
                                             q4_start,
                                             q4_start,
                                             q2_start])

        # Median
        box_plot['medians'][box_no].set_ydata([q3_start, q3_start])

        # The y axis is rescaled to fit the new box plot completely with 10%
        # of the maximum value at both ends
        axes.set_ylim([min_y * 1.1, max_y * 1.1])

    if same_color:
        plt.setp(box_plot['boxes'], color=same_color)
        plt.setp(box_plot['whiskers'], color=same_color)
        plt.setp(box_plot['medians'], color=same_color)

    # If redraw is set to true, the canvas is updated.
    if redraw:
        fig.canvas.draw()


def composite_analysis(data, col_names, label, q=[5, 25, 50, 75, 95],
                       plot_mean=True):
    data['hours'] = data.index.hour
    composite_results = {}
    fig, axes = plt.subplots(len(col_names), 1, sharex=True, dpi=150)
    for idx, col in enumerate(col_names):
        composite_results[col] = []
        perctls = []
        mean_vals = []
        counts = []
        mid_vals = []
        for h in range(0, 24, 1):
            x = data[col][(data['hours'] == h) & ~np.isnan(data[col])]
            composite_results[col].append(x)
            if len(x) > 0:
                perctls.append((np.percentile(x, q[0]), np.percentile(x, q[1]),
                                np.percentile(x, q[2]), np.percentile(x, q[3]),
                                np.percentile(x, q[4])))
            else:
                perctls.append((np.nan, np.nan, np.nan, np.nan, np.nan))
            mean_vals.append(np.nanmean(x))
            counts.append(len(x))
            mid_vals.append(np.median(x))

        top = np.nanmax(np.array(perctls))
        bot = np.nanmin(np.array(perctls))
        top = top + (top - bot) * 0.25
        bot = bot - (top - bot) * 0.05

        customized_box_plot(perctls, fig, axes=axes[idx], same_color='black')
        axes[idx].plot(range(1, 25), mean_vals, 'r.-')
        axes[idx].set_ylabel(label[idx])
        axes[idx].set_ylim([bot, top])
        for h, count, mid_val, mean_val in zip(range(1, 25), counts, mid_vals, \
                                               mean_vals):
            axes[idx].text(h, top - 0.08 * (top - bot), str(count),
                           horizontalalignment='center',
                           size='x-small', color='blue')
            axes[idx].text(h, top - 0.13 * (top - bot), str(np.round(mid_val, 1)),
                           horizontalalignment='center',
                           size='x-small', color='black')
            axes[idx].text(h, top - 0.18 * (top - bot), str(np.round(mean_val, 1)),
                           horizontalalignment='center',
                           size='x-small', color='red')
        axes[idx].text(-1.5, top - 0.08 * (top - bot), 'Count',
                       horizontalalignment='center',
                       size='x-small', color='blue')
        axes[idx].text(-1.5, top - 0.13 * (top - bot), 'Median',
                       horizontalalignment='center',
                       size='x-small', color='black')
        axes[idx].text(-1.5, top - 0.18 * (top - bot), 'Mean',
                       horizontalalignment='center',
                       size='x-small', color='red')

    axes[idx].set_xlabel('Local time (hour)')
    fig.tight_layout()
    plt.show()


def fill_null(target_gas):
    """
    Fill np.nan to tur_flux, air_prop, cospectra, sst, itc for combine final
    output.

    Returns
    -------
    tur_flux: dict
        Turbulent flux results.
    air_prop: dict
        Air properties.
    gas_den_avg: dict
        Gas density averaged for 30 min.
    cospectra: dict
        Cospectra output.
    sst: dict
        Stationarity test results.
    itc: dict
        Integrated turbulence test results.

    """
    tur_flux = {'tau': np.nan, 'H': np.nan, 'LE': np.nan, 'F_CO2': np.nan,
                'F_H2O': np.nan, 'F_' + target_gas: np.nan, 'w_WPL': np.nan,
                'TKE': np.nan, 'L': np.nan, 'u_star': np.nan,
                'wind_speed': np.nan,
                'wind_direction': np.nan}
    air_prop = {'T_avg': np.nan, 'rho_air': np.nan, 'q_avg': np.nan,
                'p_avg': np.nan, 'e': np.nan, 'es': np.nan, 'RH': np.nan,
                'theta_v_avg': np.nan}
    gas_den_avg = {target_gas: np.nan, 'H2O': np.nan, 'CO2': np.nan}
    itc = np.nan
    sst = {'wT': np.nan, 'wH2O': np.nan, 'wCO2': np.nan, 'w' + target_gas:
        np.nan}
    nan_cosp = np.full((64,), np.nan)
    cospectra = {'wT': nan_cosp, 'wH2O': nan_cosp, 'wCO2': nan_cosp,
                 'w' + target_gas: nan_cosp, 'L': np.nan, 'u_star': np.nan}

    return tur_flux, air_prop, gas_den_avg, cospectra, sst, itc


def main():
    print 'ECTool is a package for eddy covariance method.'


main()
