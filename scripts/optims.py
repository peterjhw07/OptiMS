"""Automated optimisation and/or parameter changing program - designed for use with Waters MassLynx"""

from datetime import datetime
from itertools import product
import numpy as np
import os.path
import pandas as pd
import pickle
import PIL
import pyautogui as pg
from skopt import gp_minimize  # install as scikit-optimize module
import sys
import time
import random
# import timeit
import warnings
import func


# These first four functions only apply if optimising mass spectrum parameters
# Input custom metric, if required
def custom_metric(avg_data):
    return avg_data[1]  # print('Design custom metric')


# If using a custom software, input any sequence required for refreshing the chromatogram
def chrom_refresh_custom(other_coord):
    return None


# If using a custom software, input any lines required to get the chromatogram data into a pandas.DataFrame format
# with headers 'Time' and 'Intensity'
def get_chrom_custom(chrom_coord, other_coord):
    func.left_click(chrom_coord)
    func.left_click(other_coord[0])
    data = pd.read_clipboard(header=None)
    data.columns = ['Time', 'Intensity']
    return data


# If using a custom software, input any sequence required for stopping the aquisition
def stop_aq_custom(stop_coord, other_coord):
    func.left_click(stop_coord)
    return None


# Main OptiMS program
def run_optims(param_names, default_params=None, param_bounds=None, tab_names=None, tab_rows=None, param_in_tab=None,
               method_type='hone', method_metric='max', break_fac=0, chrom_num=1, other_coord_num=0,
               param_change_delay=10, stabil_delay=2, hold_end=10, n_random_points=60, n_honing_points=60,
               software=None, learn_coord=True, pic_folder=None, output_file=None, get_csv=False, get_Excel=False):
    """
    Alters and optimises instrument conditions as specified

    Params
    ------
    param_names : list of str
        Names of desired variable instrument parameters
    default_params : list of float or None, optional
        Default (or starting) values of param_names. Default is None
    param_bounds : list of tuple of float, str or None, optional
        Lower, upper and step-size of parameter bounds of param_names, in the form (lower, upper, step-size)
        (all method_type except defined), or input params in form
        [(param 1 value 1, param 2 value 1, ...), (param 1 value 2, param 2 value 2, ...), ...]
        or str of previously exported '_optims_output.xlsx' file (both defined method_type only). Default is None
    tab_names : list of str or None, optional
        Names of tabs which param_names are located under. Optional if all param_names under single tab
    tab_rows : list of int or None, optional
        Row locations of tab_names, i.e. if tabs are on two sets of rows, input tab_names location as e.g. [1, 1, 2, ...]
    param_in_tab : list of int or None, optional
        Locations of param_names under tab_names, e.g. [1, 1, 2, ...]
    method_type : str
        Method for altering and optimising instrument conditions. Methods are:
            exhaust - runs all possible combinations of parameters, as specified by param_bounds;
            random - runs random combinations of parameters, as specified by param_bounds;
            defined - runs set combinations of parameters, as specified by param_bounds
                        or input_param if specified;
            simple - optimises instrument conditions by sequentially optimising each parameters,
                        as specified by param_bounds;
            hone - optimises instrument conditions by first running random combination of parameters,
                    as specified by param_bounds, then choosing the best combination
                    and then making minor alterations to parameters to hone in on best solution
    method_metric : str
        Metric for determining optimal conditions. Default is 'max'. Metrics are:
            max - maximises intensity of chromatogram 1. Default;
            max_2 - maximises intensity of chromatogram 2;
            sum - sums intensities of all chromatograms;
            sum_2 - sums intensities of second chromatogram to last chromatogram;
            ratio - maximises ratio of chromatogram 1 to chromatogram 2;
            custom - metric defined by the user
    break_fac : float
        Factor only for the 'simple' method, for which the method will break if the new set of parameters
        with the new method_metric < break_fac * old method_metric. Default is 0, i.e. never breaks
    chrom_num : int
        Number of chromatograms desired to be recorded. Default is 1.
        If chrom_num = 0, no chromatogram and hence chromatogram averages, errors or metrics will be recorded,
        hence is not compatible with simple or hone method_type.
    param_change_delay : float
        Time between parameter changes in seconds. Default is 10
    stabil_delay : float
        Time required for instrument to stabilise, following parameter changes in seconds. Default is 2
    hold_end : float
        Time for which determined optimised conditions are held at the end of optimisation in seconds.
        If chrom_num = 0, hold_end does not apply. Default is 60
    n_random_points : int
        Number of random points for use in random and hone method_type only. Default is 60
    n_hone_points : int
        Number of honing points for use in hone method_type only. Default is 60
    learn_coord : bool
        Specifies whether on-screen coordinates need to be learnt. Default is True
    software : str, optional
        Input software. Currently accepted softwares are MassLynx (Waters), Xcalibur (Thermo) and custom (define above).
        Default is None, which is not compatible with simple or hone method_type.
    pic_folder : str or None, optional
        Folder for exportation of screen grabbed pictures, for easier data processing.
        Default is None (screen grabs not taken)
    output_file : str or None, optional
        Location and name of output file for exportation of input parameters and results.
        Do not include a file extension. Default is None (no file exportation)
    get_csv : bool, optional
        Determines whether DataFrames will outputted as csv file. Default is False
    get_excel : bool, optional
        Determines whether DataFrames will outputted as Microsoft Excel file. Default is False

    Returns
    -------
    opti_store_df : pandas.DataFrame
        DataFrame of recorded times following instrument parameter alterations, instrument parameter alterations,
        and recorded averages, errors and metrics if chrom_num > 0
    opti_param : list
        Optimised parameters. Returns 'Unknown' if chrom_num = 0
    """

    def param_list(bounds):
        return np.linspace(bounds[0], bounds[1], int(abs(bounds[1] - bounds[0]) / bounds[2]) + 1)

    class find_tab_row:
        last_i = 1

        def get_last_i(self):
            return self.last_i

        def replace_last_i(self, curr_i):
            self.last_i = curr_i

    def change_all_params(param_names, tab_coord, param_in_tab, tab_rows, param_coord, params):
        for i in range(len(param_names)):
            left_click_tab(tab_coord[param_in_tab[i]], tab_rows[param_in_tab[i]])
            func.left_click_enter_param(param_coord[i], params[i])
        return

    def simple(headers, tab_coord, tab_rows, data_store_filename, param_names, exp_start_time, chrom_coord, other_coord,
               snip_screen_coord, bounds, coord, start_param, param_name, param_change_delay, stabil_delay, break_fac):
        opti_store_df = pd.read_pickle(data_store_filename)
        cycle_start = get_chrom_curr_time(chrom_coord, other_coord)  # Records time of start of parameter cycle
        left_click_tab(tab_coord, tab_rows)
        metric_prev = 0
        for i in param_list(bounds):
            start_param[param_names.index(param_name)] = i
            func.left_click_enter_param(coord, i)
            opti_store_df, metric_recent = sleep_avg_store(opti_store_df, data_store_filename, headers, exp_start_time,
                                                           start_param, chrom_coord, other_coord, snip_screen_coord,
                                                           param_change_delay, stabil_delay)
            if metric_recent < break_fac * metric_prev:
                break
            metric_prev = metric_recent
        opti_store_df_cycle = opti_store_df.loc[opti_store_df['Chrom time'] >= cycle_start]
        opti_param = opti_store_df_cycle[param_names].loc[opti_store_df_cycle['Metric'].astype(float).idxmax()]
        func.left_click_enter_param(coord, opti_param[param_name])
        return opti_param

    def hone(params, hone_factors, data_store_filename, param_names, exp_start_time, tab_coord, param_in_tab, tab_rows,
             param_coord, chrom_coord, other_coord, snip_screen_coord, param_change_delay, stabil_delay, headers):
        params_refac = [i * j for i, j in zip(params, hone_factors)]
        opti_store_df = pd.read_pickle(data_store_filename)
        if opti_store_df.shape[0] == 0 or not (opti_store_df[param_names].iloc[-1:] == np.array(params_refac)).all(
                1).any():
            change_all_params(param_names, tab_coord, param_in_tab, tab_rows, param_coord, params_refac)
            opti_store_df, metric_recent = sleep_avg_store(opti_store_df, data_store_filename, headers, exp_start_time,
                                                           params_refac, chrom_coord, other_coord, snip_screen_coord,
                                                           param_change_delay, stabil_delay)
        else:
            found_row = opti_store_df[
                (opti_store_df.loc[:, param_names] == np.array(params_refac)).all(1)].index.tolist()
            metric_recent = opti_store_df.loc[found_row[0], 'Metric']
        return -metric_recent

    def end_hold(opti_store_df, data_store_filename, headers, exp_start_time, start_param, chrom_coord, other_coord,
                 snip_screen_coord, param_change_delay, stabil_delay):
        opti_store_df, metric_recent = sleep_avg_store(opti_store_df, data_store_filename, headers, exp_start_time,
                                                       start_param, chrom_coord, other_coord, snip_screen_coord,
                                                       param_change_delay, stabil_delay)
        return opti_store_df

    if 'MassLynx' in software:
        other_coord_num = 1

        def chrom_refresh(other_coord):
            return None

        def get_chrom(chrom_coord, other_coord):
            func.left_click(chrom_coord)
            func.left_click(other_coord[0])
            data = pd.read_clipboard(header=None)
            # data = pd.DataFrame(pd.read_clipboard(header=None), columns=['Time', 'Intensity'])
            data.columns = ['Time', 'Intensity']
            return data

        def stop_aq(stop_coord, other_coord):
            func.left_click(stop_coord)
            pg.typewrite(['enter'])

    if 'Xcalibur' in software:
        other_coord_num = 0

        def chrom_refresh(other_coord):
            pg.typewrite(['F5'])

        def get_chrom(chrom_coord, other_coord):
            pg.click(chrom_coord[0], chrom_coord[1], button='right')
            pg.click(chrom_coord[0] + 10, chrom_coord[1] + 50)
            pg.click(chrom_coord[0] + 20, chrom_coord[1] + 50)
            data = pd.read_clipboard(header=3)
            data.columns = ['Time', 'Intensity']
            return data

        def stop_aq(stop_coord, other_coord):
            func.left_click(stop_coord)

    if 'custom' in software:

        def chrom_refresh(other_coord):
            chrom_refresh_custom(other_coord)

        def get_chrom(chrom_coord, other_coord):
            return get_chrom_custom(chrom_coord, other_coord)

        def stop_aq(stop_coord, other_coord):
            stop_aq_custom(stop_coord, other_coord)

    def get_chrom_curr_time(chrom_coord, other_coord):
        chrom_refresh(other_coord)
        return get_chrom(chrom_coord[0], other_coord)['Time'].iat[-1]  # df['Time'].iat[-1]

    if chrom_num == 0:
        def get_metric(avg_data):
            return 'Unknown'
    elif 'max' not in method_metric and 'ratio' not in method_metric and 'custom' not in method_metric:
        method_metric = input('Invalid method_metric given. Enter valid method_metric \n')
    if 'max' in method_metric:
        def get_metric(avg_data):
            return avg_data[0]
    elif 'max_2' in method_metric:
        def get_metric(avg_data):
            return avg_data[1]
    elif 'sum' in method_metric:
        def get_metric(avg_data):
            return sum(avg_data)
    elif 'sum_2' in method_metric:
        def get_metric(avg_data):
            return sum(avg_data[1:])
    elif 'ratio' in method_metric:
        def get_metric(avg_data):
            return avg_data[0] / avg_data[1]
    elif 'custom' in method_metric:
        def get_metric(avg_data):
            return custom_metric(avg_data)

    def get_avg_chrom(chrom_coord, other_coord, param_change_delay, stabil_delay):
        avg, error, error_perc = [], [], []
        curr_time = get_chrom_curr_time(chrom_coord, other_coord)
        for i in range(len(chrom_coord)):
            df = get_chrom(chrom_coord[i], other_coord)
            df_loc = df.loc[(df['Time'] <= curr_time) &
                            (df['Time'] >= (curr_time - (param_change_delay / 60) + stabil_delay))]
            avg.append(df_loc['Intensity'].mean())
            error.append(df_loc['Intensity'].std() / (len(df_loc['Intensity']) ** 0.5))
            error_perc.append((error[-1] / avg[-1]) * 100)
        return avg, error, error_perc

    if chrom_num != 0 and software:
        def sleep_avg_store(opti_store_df, data_store_filename, headers, exp_start_time, params, chrom_coord,
                            other_coord, snip_screen_coord, param_change_delay, stabil_delay):
            chrom_start_time = get_chrom_curr_time(chrom_coord, other_coord)
            time.sleep(param_change_delay)
            snip_screen(chrom_start_time, snip_screen_coord)
            chrom_avg, chrom_error, chrom_error_perc = get_avg_chrom(chrom_coord, other_coord, param_change_delay,
                                                                     stabil_delay)
            metric_recent = get_metric(chrom_avg)
            opti_store_df = pd.concat([opti_store_df, pd.DataFrame(
                np.array([[chrom_start_time, *params, chrom_avg, chrom_error, chrom_error_perc,
                           metric_recent]]), columns=headers)], ignore_index=True)
            opti_store_df.to_pickle(data_store_filename)
            print(opti_store_df)
            return opti_store_df, metric_recent
    else:
        def sleep_avg_store(opti_store_df, data_store_filename, headers, exp_start_time, params, chrom_coord,
                            other_coord, snip_screen_coord, param_change_delay, stabil_delay):
            chrom_start_time = datetime.now().timestamp()  # - exp_start_time
            # time.sleep(max(0, param_change_delay - (
            # (datetime.now().timestamp() - exp_start_time) - (opti_store_df.shape[0] * param_change_delay))))
            time.sleep(param_change_delay)
            snip_screen(chrom_start_time, snip_screen_coord)
            opti_store_df = pd.concat([opti_store_df, pd.DataFrame(
                np.array([[chrom_start_time, *params]]), columns=headers)], ignore_index=True)
            opti_store_df.to_pickle(data_store_filename)
            print(opti_store_df)
            return opti_store_df, 'Unknown'

    if not tab_names:
        def left_click_tab(tab_coord, tab_row):
            return None
    elif not tab_rows or min(tab_rows) == max(tab_rows):
        def left_click_tab(tab_coord, tab_row):
            func.left_click(tab_coord)
    else:
        def left_click_tab(tab_coord, tab_row):
            if tab_row == find_tab_row_1.get_last_i():
                func.left_click((tab_coord[0], tab_y_lower))
            else:
                func.left_click((tab_coord[0], tab_y_upper))
            find_tab_row_1.replace_last_i(tab_row)

    if pic_folder and os.path.exists(pic_folder):
        def snip_screen(chrom_time, snip_screen_coord):
            image = PIL.ImageGrab.grab(bbox=(
                snip_screen_coord[0][0], snip_screen_coord[0][1], snip_screen_coord[1][0], snip_screen_coord[1][1]),
                all_screens=True)
            image.save(pic_folder + r'\OptiMS_screenshot_' + str(chrom_time) + '.png')
    else:
        def snip_screen(time, snip_screen_coord):
            return

    coord_store_filename = 'coord_store_file.pkl'
    data_store_filename = 'data_store_file.pkl'
    param_store_filename = 'param_store_file.pkl'

    def chrom_copy_error():
        return func.gen_err('Suggested error! Cannot copy chromatogram. Set chrom_num=0 or check coordinates and ensure chromatogram is visible.')

    def other_error():
        return func.gen_err('Unknown error! Check inputs are formatted correctly. Else examine error messages and review code.')

    if any(i in method_type for i in ('simple', 'hone')) and chrom_num == 0:
        print('Simple and hone method types are not compatible with chrom_num = 0. Please select a different method.')
        sys.exit()

    if learn_coord:
        tab_coord, param_coord, chrom_coord, other_coord, snip_screen_coord = [], [], [], [], []

        tab_coord = [func.coord_find(i + ' tab') for i in tab_names]
        param_coord = [func.coord_find(i + ' box') for i in param_names]
        chrom_coord = [func.coord_find('chromatogram ' + str(i + 1)) for i in range(chrom_num)]
        if chrom_num != 0 and 'MassLynx' in software:
            other_coord = [func.coord_find('copy chromatogram button')]
        else:
            other_coord = [func.coord_find('other coordinate ' + str(i + 1)) for i in range(other_coord_num)]
        if os.path.exists(pic_folder):
            snip_screen_coord.append(func.coord_find('first corner of snip window'))
            snip_screen_coord.append(func.coord_find('opposite corner of snip window'))
        stop_coord = func.coord_find('Stop button')
        all_coord = tab_coord, param_coord, chrom_coord, other_coord, snip_screen_coord, stop_coord
        with open(coord_store_filename, 'wb') as outp:
            pickle.dump(all_coord, outp, pickle.HIGHEST_PROTOCOL)

    with open(coord_store_filename, 'rb') as inp:
        all_coord = pickle.load(inp)
    tab_coord, param_coord, chrom_coord, other_coord, snip_screen_coord, stop_coord = all_coord

    if not tab_names:
        tab_names = [None]
        tab_coord = [(0, 0)]
    if not param_in_tab:
        param_in_tab = [1] * len(param_names)
    if tab_rows and min(tab_rows) != max(tab_rows):
        tabs_x, tabs_y = zip(*tab_coord)
        tab_y_upper = np.mean(tabs_y[tab_rows.index(1)])
        tab_y_lower = np.mean(tabs_y[tab_rows.index(2)])
    else:
        tab_rows = [1] * len(tab_names)

    if not 'defined' in method_type and not len(param_names) == len(default_params) == len(param_bounds) == len(param_in_tab):
        print('Error! Mismatch between number of param_names, default_params, param_bounds and param_in_tab.')
        sys.exit(1)

    param_store_df = pd.DataFrame({'Parameter': ['param_names', 'default_params', 'param_bounds', 'tab_names',
                                                 'tab_rows', 'param_in_tab', 'method_type', 'method_metric',
                                                 'break_fac', 'chrom_num', 'param_change_delay', 'stabil_delay',
                                                 'hold_end', 'n_random_points', 'n_honing_points'],
                                   'Values': [param_names, default_params, param_bounds, tab_names,
                                              tab_rows, param_in_tab, method_type, method_metric,
                                              break_fac, chrom_num, param_change_delay, stabil_delay,
                                              hold_end, n_random_points, n_honing_points]})

    if chrom_num != 0:
        headers = ['Chrom time', *param_names, 'Chrom averages', 'Chrom errors', 'Chrom errors %', 'Metric']
    else:
        headers = ['Chrom time', *param_names]
    tab_rows = [i - 1 for i in tab_rows]
    param_in_tab = [i - 1 for i in param_in_tab]
    stabil_delay = stabil_delay / 60

    find_tab_row_1 = find_tab_row()

    method_type_try = False
    while method_type_try is False:
        if any(i in method_type for i in ('defined', 'exhaust', 'random', 'simple', 'hone', 'recover')):
            method_type_try = True
        else:
            method_type = input('Error! Invalid method_type. Enter valid method_type. \n')

    if 'exhaust' in method_type:
        print('Calculating all parameter combinations. This may take some time.')
        all_param_val = []
        for i in param_bounds:
            all_param_val.append(param_list(i))
        param_combi = list(product(*all_param_val))
        param_store_df = pd.concat([param_store_df, pd.DataFrame({'Parameter': ['param_combi'],
                                                                  'Values': [param_combi]})], ignore_index=True)
    elif 'random' in method_type:
        print('Calculating all parameter combinations and choosing random combinations. This may take some time.')
        rand_combi = []
        for i in param_bounds:
            rand_combi.append(random.choices(np.ndarray.tolist(param_list(i)), k=n_random_points))
        param_combi = tuple(zip(*rand_combi))
        param_store_df = pd.concat([param_store_df, pd.DataFrame({'Parameter': ['param_combi'],
                                                                  'Values': [param_combi]})], ignore_index=True)
    elif 'defined' in method_type:
        if isinstance(param_bounds, list):
            param_combi = param_bounds
        elif isinstance(param_bounds, str):
            param_store_df = pd.read_excel(param_bounds, sheet_name='Params')
            param_combi = param_store_df.at[param_store_df.index[param_store_df['Parameter'] == 'param_combi'].tolist()[0],
                                        'Values']
            param_combi = eval(param_combi)
        param_store_df = pd.concat([param_store_df, pd.DataFrame({'Parameter': ['param_combi'],
                                                                  'Values': [param_combi]})], ignore_index=True)

    if any(i in method_type for i in ('exhaust', 'random', 'defined')):
        if isinstance(param_change_delay, (int, float)):
            param_change_delay = [param_change_delay] * len(param_combi)
        opti_store_df = pd.DataFrame(columns=headers)
        opti_store_df.to_pickle(data_store_filename)
        exp_start_time = datetime.now().timestamp()
        try:
            for i in range(len(param_combi)):
                change_all_params(param_names, tab_coord, param_in_tab, tab_rows, param_coord, param_combi[i])
                opti_store_df, metric_recent = sleep_avg_store(opti_store_df, data_store_filename, headers,
                                                               exp_start_time, param_combi[i], chrom_coord, other_coord,
                                                               snip_screen_coord, param_change_delay[i], stabil_delay)
        except IndexError:
            chrom_copy_error()
        except Exception:
            other_error()
        if chrom_num != 0:
            opti_param = opti_store_df[param_names].loc[opti_store_df['Metric'].astype(float).idxmax()]  # added .astype(float) to cope with different df style
            change_all_params(param_names, tab_coord, param_in_tab, tab_rows, param_coord, opti_param)
        else:
            opti_param = 'Unknown'

    elif 'simple' in method_type:
        start_param = default_params
        opti_store_df = pd.DataFrame(columns=headers)
        opti_store_df.to_pickle(data_store_filename)
        exp_start_time = datetime.now().timestamp()
        change_all_params(param_names, tab_coord, param_in_tab, tab_rows, param_coord, default_params)
        try:
            for i in range(len(param_names)):
                start_param = simple(headers, tab_coord[param_in_tab[i]], tab_rows[param_in_tab[i]],
                                     data_store_filename, param_names, exp_start_time, chrom_coord, other_coord,
                                     snip_screen_coord, param_bounds[i], param_coord[i], start_param, param_names[i],
                                     param_change_delay, stabil_delay, break_fac)
        except IndexError:
            chrom_copy_error()
        except Exception:
            other_error()

        opti_store_df = pd.read_pickle(data_store_filename)
        opti_param = start_param

    elif 'hone' in method_type:
        opti_store_df = pd.DataFrame(columns=headers)
        opti_store_df.to_pickle(data_store_filename)
        exp_start_time = datetime.now().timestamp()
        hone_ranges, hone_defaults, hone_factors = [], [], []
        warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')
        warnings.filterwarnings('ignore',
                                message='string or file could not be read to its end due to unmatched data; '
                                        'this will raise a ValueError in the future.')
        for i in range(len(param_names)):
            hone_ranges.append(
                [int(param_bounds[i][0] / param_bounds[i][2]), int(param_bounds[i][1] / param_bounds[i][2])])
            hone_defaults.append(int(default_params[i] / param_bounds[i][2]))
            hone_factors.append(param_bounds[i][2])
        lam_hone = lambda param_tup: hone(param_tup, hone_factors, data_store_filename, param_names, exp_start_time,
                                          tab_coord, param_in_tab, tab_rows, param_coord, chrom_coord, other_coord,
                                          snip_screen_coord, param_change_delay, stabil_delay, headers)
        try:
            res_gp = gp_minimize(lam_hone, hone_ranges, x0=hone_defaults, n_initial_points=n_random_points,
                                 n_calls=n_random_points + max(0, n_honing_points))
        except IndexError:
            chrom_copy_error()
        except Exception:
            other_error()
        opti_store_df = pd.read_pickle(data_store_filename)
        opti_param = [float(res_gp.x[i]) * param_bounds[i][2] for i in range(len(param_names))]
        change_all_params(param_names, tab_coord, param_in_tab, tab_rows, param_coord, opti_param)

    if any(i in method_type for i in ('defined', 'exhaust', 'random', 'simple', 'hone')):
        opti_store_df = end_hold(opti_store_df, data_store_filename, headers, exp_start_time, opti_param, chrom_coord,
                                 other_coord, snip_screen_coord, hold_end, stabil_delay)
        stop_aq(stop_coord, other_coord)

    if 'recover' in method_type:
        param_store_df = pd.read_pickle(param_store_filename)
        opti_store_df = pd.read_pickle(data_store_filename)
        if chrom_num != 0:
            try:
                opti_param = opti_store_df[param_names].loc[opti_store_df['Metric'].astype(float).idxmax()]
            except IndexError:
                chrom_copy_error()
            except Exception:
                other_error()
        else:
            opti_param = 'Unknown'

    if get_csv and os.path.exists(output_file + '_optims_output.txt'):
        print('WARNING! ' + output_file + '_optims_output.txt already exists and will be overwritten.')
    if get_Excel and os.path.exists(output_file + '_optims_output.xlsx'):
        print('WARNING! ' + output_file + '_optims_output.xlsx already exists and will be overwritten.')
    if (get_csv and os.path.exists(output_file + '_optims_output.txt')) or (get_Excel and os.path.exists(output_file + '_optims_output.xlsx')):
        print('Press enter to continue.')
        input()
    if get_csv:
        opti_store_df.to_csv(output_file + '_optims_output.txt', header=headers, index=None, sep=' ', mode='w')
    if get_Excel:
        func.save_excel([param_store_df, opti_store_df], ['Params', 'Output'], output_file + '_optims_output')

    # print(opti_store_df)
    # print(opti_param)

    return opti_store_df, opti_param


if __name__ == "__main__":
    # Input optimisation method details
    method_type = 'random'  # input method type: 'defined' for chosen parameter values; 'exhaust' for exhaustive; 'random' for random combination from exhaustive; 'simple' for simple; 'hone' for honing; 'recover' to export last data set
    method_metric = 'max'  # input metric defition: 'max' for maximising intensity of first chromatogram; 'ratio' for maximising intensity of first chromatogram to second; 'custom' to define custom metric (below)
    method_break = 0  # input a factor for which the method will break and move onto the next parameter, if the new metric falls below method_break * previous (simple only).
    n_random_points = 5  # input number of random combinations of parameters to guess, must be >0 (random and hone only).
    n_honing_points = 5  # input number of subsequent honing points required, must be >0 (hone only).

    # Input parameter details, including bounds
    tab_names = ['ES', 'Instrument']  # input names of param_in_tab in form ['Tab 1', ..., 'Tab X'] or None or [] if only tab used.
    tab_rows = []  # input row of each tab in form [1, ..., 2] or None or [] if all param_in_tab on same row. Only two rows are currently supported.
    param_names = ['Capillary', 'Sampling Cone', 'Source Offset', 'Nebuliser', 'TrapCE', 'TransCE']  # input names of parameters in form ['Param 1, ..., Param X']
    default_params = [3, 0, 50, 3, 5, 0]  # input default start values in the form [0, ..., 0].
    param_bounds = [(0, 5, 1), (0, 100, 10), (0, 100, 10), (2.5, 6, 0.5), (0, 10, 1), (0, 10, 1)]  # input parameters bounds (lower bound, upper bound, increment value) in form e.g. [(X1, X2, X3), ..., (Y1, Y2, Y3)] (all methods except defined) or input params in form [(param 1 value 1, param 2 value 1, ...), (param 1 value 2, param 2 value 2, ...), ...] or location of previously saved parameter details (defined only).
    param_in_tab = []  # enter param_in_tab required for each parameter in form [1, ..., X] or None or [] if all parameters in Tab 1.
    chrom_num = 0  # enter number of chromatograms in use. Input 0 if chromatograms cannot be copied. Else input integer > 0.
    other_coord_num = 1

    learn_coord = True  # input True if wanting to learn coordinates of required param_in_tab/boxes/chromatograms, else False.

    # Input system times
    param_change_delay = 5  # input delay between change of parameters in seconds
    stabil_delay = 2  # input time taken for system to stabilise in seconds
    hold_end = 10  # input time desired to hold optimal parameters immediately before termination in seconds

    # Input output file/folder locations
    software = 'custom'  # input software in use. Currently accepted softwares are MassLynx (Waters), Xcalibur (Thermo) and custom (define above)
    pic_folder = r''  # input folder location for saving pictures as r'folder_location\folder_name' if desired - else put False or r''
    output_file = r'C:\Users\Waters\Documents\OptiMS\Exp'  # input location of optimisation file output as r'file_location\name' - do not include file extension, e.g. C:\Users\Waters\Documents\OptiMS\Exp.

    opti_param, opti_store_df = run_optims(param_names, default_params=default_params, param_bounds=param_bounds,
                                           tab_names=tab_names, tab_rows=tab_rows, param_in_tab=param_in_tab,
                                           method_type=method_type, method_metric=method_metric, break_fac=method_break,
                                           chrom_num=chrom_num, other_coord_num=other_coord_num,
                                           param_change_delay=param_change_delay, stabil_delay=stabil_delay,
                                           hold_end=hold_end, n_random_points=n_random_points,
                                           n_honing_points=n_honing_points, software=software, learn_coord=learn_coord,
                                           pic_folder=pic_folder, output_file=output_file,
                                           get_csv=False, get_Excel=True)
