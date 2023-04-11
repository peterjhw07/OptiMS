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
import traceback
import warnings
import master_grab as mg


def param_list(bounds):
    return np.linspace(bounds[0], bounds[1], int(abs(bounds[1] - bounds[0]) / bounds[2]) + 1)


def get_chrom(chrom_coord, copy_coord):
    pg.click(chrom_coord[0], chrom_coord[1], clicks=1, button='left')
    pg.click(copy_coord[0], copy_coord[1], clicks=1, button='left')
    return pd.DataFrame(pd.read_clipboard(header=None), columns=['Time', 'Intensity'])


def get_chrom_curr_time(chrom_coord, copy_coord):
    df = get_chrom(chrom_coord[0], copy_coord)
    return df['Time'].iat[-1]


def get_avg_chrom(chrom_coord, copy_coord, param_change_delay, stabil_delay):
    df = pd.DataFrame(columns=['Time', 'Intensity'])
    avg, error, error_perc = [], [], []
    for i in range(len(chrom_coord)):
        df_temp = get_chrom(chrom_coord[i], copy_coord)
        curr_time = df_temp['Time'].iat[-1]
        df_temp_loc = df_temp.loc[
            df_temp['Time'] >= (curr_time - (param_change_delay / 60) + stabil_delay)]
        avg.append(df_temp_loc['Intensity'].mean())
        error.append(df_temp_loc['Intensity'].std() / (len(df_temp_loc['Intensity']) ** 0.5))
        error_perc.append((error[-1] / avg[-1]) * 100)
        df = pd.concat([df, df_temp], axis=1)
    return avg, error, error_perc


class find_tab_row:
    last_i = 1

    def get_last_i(self):
        return self.last_i

    def replace_last_i(self, curr_i):
        self.last_i = curr_i


def gen_err(text):
    print(traceback.format_exc())
    print(text)
    sys.exit(1)


def run_optims(param_names, default_params, param_bounds, tab_names=None, tab_rows=None, param_in_tab=None,
               method_type='hone', method_metric='max', break_fac=0, chrom_num=1,
               param_change_delay=10, stabil_delay=2, hold_end=10, n_random_points=60, n_honing_points=60,
               learn_coord=True, output_file=None, input_file=None, pic_folder=None):
    """
    Alters and optimises instrument conditions as specified

    Params
    ------
    param_names : list of str
        Names of desired variable instrument parameters
    default_params : list of float
        Default (or starting) values of param_names
    param_bounds : list of tuple of float
        Lower, upper and step-size of parameter bounds of param_names, in the form (lower, upper, step-size)
        or defined values in the form (param 1, param 2, ...) (defined method_type only)
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
                        or input_file if specified;
            simple - optimises instrument conditions by sequentially optimising each parameters,
                        as specified by param_bounds;
            hone - optimises instrument conditions by first running random combination of parameters,
                    as specified by param_bounds, then choosing the best combination
                    and then making minor alterations to parameters to hone in on best solution
    method_metric : str
        Metric for determining optimal conditions. Default is 'max'. Metrics are:
            max - maximises intensity of chromatogram 1. Default;
            max_2 - maximises intensity of chromatogram 2;
            ratio - maximises ratio of chromatogram 1 to chromatogram 2;
            user def - metric defined by the user
    break_fac : float
        Factor only for the 'simple' method, for which the method will break if the new set of parameters
        with the new method_metric < break_fac * old method_metric. Default is 0, i.e. never breaks
    chrom_num : int
        Number of chromatograms desired to be recorded. Default is 1.
        If chrom_num = 0, no chromatogram and hence chromatogram averages, errors or metrics will be recorded.
        Not compatible with simple or hone method_type.
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
    output_file : str or None, optional
        Location and name of output file for exportation of input parameters and results.
        Do not include a file extension. Default is None (no file exportation)
    input_file : str or None, optional
        Input Excel file for importation of input parameters and results (defined method_type only).
        Must be or adapted from a previously exported output_file. Default is None (no file importation)
    pic_folder : str or None, optional
        Folder for exportation of screen grabbed pictures, for easier data processing.
        Default is None (screen grabs not taken)

    Returns
    -------
    opti_store_df : pandas.DataFrame
        DataFrame of recorded times following instrument parameter alterations, instrument parameter alterations,
        and recorded averages, errors and metrics if chrom_num > 0
    opti_param : list
        Optimised parameters. Returns 'Unknown' if chrom_num = 0
    """

    def change_all_params(param_names, tab_coord, param_in_tab, tab_rows, param_coord, params):
        for i in range(len(param_names)):
            left_click_tab(tab_coord[param_in_tab[i]], tab_rows[param_in_tab[i]])
            mg.left_click_enter_param(param_coord[i], params[i])
        return

    def simple(headers, tab_coord, tab_rows, data_store_filename, param_names, exp_start_time, chrom_coord, copy_coord,
               snip_screen_coord, bounds, coord, start_param, param_name, param_change_delay, stabil_delay, break_fac):
        opti_store_df = pd.read_pickle(data_store_filename)
        cycle_start = get_chrom_curr_time(chrom_coord, copy_coord)
        left_click_tab(tab_coord, tab_rows)
        metric_prev = 0
        for i in param_list(bounds):
            start_param[param_names.index(param_name)] = i
            mg.left_click_enter_param(coord, i)
            opti_store_df, metric_recent = sleep_avg_store(opti_store_df, data_store_filename, headers, exp_start_time,
                                                           start_param, chrom_coord, copy_coord, snip_screen_coord,
                                                           param_change_delay, stabil_delay)
            if metric_recent < break_fac * metric_prev:
                break
            metric_prev = metric_recent
        opti_store_df_cycle = opti_store_df.loc[opti_store_df['Chrom time'] >= cycle_start]
        opti_param = opti_store_df_cycle[param_names].loc[opti_store_df_cycle['Metric'].idxmax()]
        mg.left_click_enter_param(coord, opti_param[param_name])
        return opti_param

    def hone(params, hone_factors, data_store_filename, param_names, exp_start_time, tab_coord, param_in_tab, tab_rows,
             param_coord, chrom_coord, copy_coord, snip_screen_coord, param_change_delay, stabil_delay, headers):
        params_refac = [i * j for i, j in zip(params, hone_factors)]
        opti_store_df = pd.read_pickle(data_store_filename)
        if opti_store_df.shape[0] == 0 or not (opti_store_df[param_names].iloc[-1:] == np.array(params_refac)).all(
                1).any():
            change_all_params(param_names, tab_coord, param_in_tab, tab_rows, param_coord, params_refac)
            opti_store_df, metric_recent = sleep_avg_store(opti_store_df, data_store_filename, headers, exp_start_time,
                                                           params_refac, chrom_coord, copy_coord, snip_screen_coord,
                                                           param_change_delay, stabil_delay)
        else:
            found_row = opti_store_df[
                (opti_store_df.loc[:, param_names] == np.array(params_refac)).all(1)].index.tolist()
            metric_recent = opti_store_df.loc[found_row[0], 'Metric']
        return -metric_recent

    def end_hold(opti_store_df, data_store_filename, headers, exp_start_time, start_param, chrom_coord, copy_coord,
                 snip_screen_coord, param_change_delay, stabil_delay):
        opti_store_df, metric_recent = sleep_avg_store(opti_store_df, data_store_filename, headers, exp_start_time,
                                                       start_param, chrom_coord, copy_coord, snip_screen_coord,
                                                       param_change_delay, stabil_delay)
        return opti_store_df

    if chrom_num == 0:
        def get_metric(avg_data):
            return 'Unknown'
    elif 'max' not in method_metric and 'ratio' not in method_metric and 'user def' not in method_metric:
        method_metric = input('Invalid method_metric given. Enter valid method_metric \n')
    if 'max' in method_metric:
        def get_metric(avg_data):
            metric = avg_data[0]
            return metric
    elif 'max_2' in method_metric:
        def get_metric(avg_data):
            metric = avg_data[1]
            return metric
    elif 'ratio' in method_metric:
        def get_metric(avg_data):
            metric = avg_data[0] / avg_data[1]
            return metric
    elif 'user def' in method_metric:
        def get_metric(avg_data):
            metric = avg_data[1]
            return metric

    if chrom_num != 0:
        def sleep_avg_store(opti_store_df, data_store_filename, headers, exp_start_time, params, chrom_coord,
                            copy_coord, snip_screen_coord, param_change_delay, stabil_delay):
            chrom_start_time = get_chrom_curr_time(chrom_coord, copy_coord)
            time.sleep(param_change_delay)
            snip_screen(chrom_start_time, snip_screen_coord)
            #time.sleep(max(0, param_change_delay - (
            #            (datetime.now().timestamp() - exp_start_time) - (opti_store_df.shape[0] * param_change_delay))))
            chrom_avg, chrom_error, chrom_error_perc = get_avg_chrom(chrom_coord, copy_coord, param_change_delay,
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
                            copy_coord, snip_screen_coord, param_change_delay, stabil_delay):
            chrom_start_time = datetime.now().timestamp() - exp_start_time
            time.sleep(param_change_delay)
            snip_screen(chrom_start_time, snip_screen_coord)
            opti_store_df = pd.concat([opti_store_df, pd.DataFrame(
                np.array([[chrom_start_time, *params]]), columns=headers)], ignore_index=True)
            opti_store_df.to_pickle(data_store_filename)
            print(opti_store_df)
            return opti_store_df, 'Unknown'

    if not tab_names:
        def left_click_tab(tab_coord, tab_row):
            pass
    elif not tab_rows or min(tab_rows) == max(tab_rows):
        def left_click_tab(tab_coord, tab_row):
            mg.left_click(tab_coord)
    else:
        def left_click_tab(tab_coord, tab_row):
            if tab_row == find_tab_row_1.get_last_i():
                mg.left_click((tab_coord[0], tab_y_lower))
            else:
                mg.left_click((tab_coord[0], tab_y_upper))
            find_tab_row_1.replace_last_i(tab_row)

    if os.path.exists(pic_folder):
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
        return gen_err('Suggested error! Cannot copy chromatogram. Set chrom_num=0 or check coordinates and ensure chromatogram is visible.')

    def other_error():
        return gen_err('Unknown error! Check inputs are formatted correctly. Else examine error messages and review code.')

    if any(i in method_type for i in ('simple', 'hone')) and chrom_num == 0:
        print('Simple and hone method types are not compatible with chrom_num = 0. Please select a different method.')
        sys.exit()

    if os.path.exists(output_file + '.txt'):
        print('WARNING! ' + output_file + '.txt already exists and will be overwritten.')
    if os.path.exists(output_file + '.xlsx'):
        print('WARNING! ' + output_file + '.xlsx already exists and will be overwritten.')
    if os.path.exists(output_file + '.txt') or os.path.exists(output_file + '.xlsx'):
        print('Press enter to continue.')
        input()

    if learn_coord:
        tab_coord, param_coord, chrom_coord, snip_screen_coord = [], [], [], []

        for i in tab_names:
            tab_coord.append(mg.coord_find(i + ' tab'))
        for i in param_names:
            param_coord.append(mg.coord_find(i + ' box'))
        for i in range(1, chrom_num + 1):
            chrom_coord.append(mg.coord_find('chromatogram ' + str(i)))
        if chrom_num != 0:
            copy_coord = mg.coord_find('copy chromatogram button')
        else:
            copy_coord = []
        if os.path.exists(pic_folder):
            snip_screen_coord.append(mg.coord_find('first corner of snip window'))
            snip_screen_coord.append(mg.coord_find('opposite corner of snip window'))
        stop_coord = mg.coord_find('Stop button')
        all_coord = tab_coord, param_coord, chrom_coord, copy_coord, snip_screen_coord, stop_coord
        with open(coord_store_filename, 'wb') as outp:
            pickle.dump(all_coord, outp, pickle.HIGHEST_PROTOCOL)

    with open(coord_store_filename, 'rb') as inp:
        all_coord = pickle.load(inp)
    tab_coord, param_coord, chrom_coord, copy_coord, snip_screen_coord, stop_coord = all_coord

    if tab_rows and min(tab_rows) != max(tab_rows):
        tabs_x, tabs_y = zip(*tab_coord)
        tab_y_upper = np.mean(tabs_y[tab_rows.index(1)])
        tab_y_lower = np.mean(tabs_y[tab_rows.index(2)])
    else:
        tab_rows = [1] * len(tab_names)

    if not param_in_tab:
        param_in_tab = [1] * len(param_names)

    if not len(param_names) == len(default_params) == len(param_bounds) == len(param_in_tab):
        print('Error! Mismatch between number of param_names, default_params, param_bounds and param_in_tab.')
        sys.exit(1)

    param_store_df = pd.DataFrame({'Parameter': ['method_type', 'method_metric', 'tab_names', 'tab_rows',
                                                 'param_names', 'default_params', 'param_bounds', 'param_in_tab',
                                                 'chrom_num', 'param_change_delay', 'stabil_delay', 'hold_end'],
                                   'Values': [method_type, method_metric, tab_names, tab_rows,
                                              param_names, default_params, param_bounds, param_in_tab,
                                              chrom_num, param_change_delay, stabil_delay, hold_end]})

    if chrom_num != 0:
        headers = ['Chrom time', *param_names, 'Chrom averages', 'Chrom errors', 'Chrom errors %', 'Metric']
    else:
        headers = ['Chrom time', *param_names]
    tab_rows = [tab_rows_edit - 1 for tab_rows_edit in tab_rows]
    param_in_tab = [tabs_edit - 1 for tabs_edit in param_in_tab]
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
        param_store_df = pd.read_excel(input_file, sheet_name='Params')
        param_combi = param_store_df.at[param_store_df.index[param_store_df['Parameter'] == 'param_combi'].tolist()[0],
                                        'Values']
        param_combi = eval(param_combi)

    if any(i in method_type for i in ('exhaust', 'random', 'defined')):
        opti_store_df = pd.DataFrame(columns=headers)
        opti_store_df.to_pickle(data_store_filename)
        exp_start_time = datetime.now().timestamp()
        try:
            for i in range(len(param_combi)):
                change_all_params(param_names, tab_coord, param_in_tab, tab_rows, param_coord, param_combi[i])
                opti_store_df, metric_recent = sleep_avg_store(opti_store_df, data_store_filename, headers,
                                                               exp_start_time, param_combi[i], chrom_coord, copy_coord,
                                                               snip_screen_coord, param_change_delay, stabil_delay)
        except IndexError:
            chrom_copy_error()
        except Exception:
            other_error()
        if chrom_num != 0:
            opti_param = opti_store_df[param_names].loc[opti_store_df['Metric'].idxmax()]
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
                                     data_store_filename, param_names, exp_start_time, chrom_coord, copy_coord,
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
                                          tab_coord, param_in_tab, tab_rows, param_coord, chrom_coord, copy_coord,
                                          snip_screen_coord, param_change_delay, stabil_delay, headers)
        try:
            res_gp = gp_minimize(lam_hone, hone_ranges, x0=hone_defaults, n_initial_points=n_random_points,
                                 n_calls=n_random_points + max(0, n_honing_points))
        except IndexError:
            chrom_copy_error()
        except Exception:
            other_error()
        opti_store_df = pd.read_pickle(data_store_filename)
        opti_param = []
        for i in range(len(param_names)):
            opti_param.append(float(res_gp.x[i]) * param_bounds[i][2])
        change_all_params(param_names, tab_coord, param_in_tab, tab_rows, param_coord, opti_param)

    if any(i in method_type for i in ('defined', 'exhaust', 'random', 'simple', 'hone')):
        opti_store_df = end_hold(opti_store_df, data_store_filename, headers, exp_start_time, opti_param, chrom_coord,
                                 copy_coord, snip_screen_coord, hold_end, stabil_delay)
        mg.left_click(stop_coord)
        pg.typewrite(['enter'])

    if 'recover' in method_type:
        param_store_df = pd.read_pickle(param_store_filename)
        opti_store_df = pd.read_pickle(data_store_filename)
        if chrom_num != 0:
            try:
                opti_param = opti_store_df[param_names].loc[opti_store_df['Metric'].idxmax()]
            except IndexError:
                chrom_copy_error()
            except Exception:
                other_error()
        else:
            opti_param = 'Unknown'

    opti_store_df.to_csv(output_file + '.txt', header=headers, index=None, sep=' ', mode='w')
    mg.save_excel([param_store_df, opti_store_df], ['Params', 'Output'], output_file)


    # print(opti_store_df)
    # print(opti_param)

    return opti_store_df, opti_param


if __name__ == "__main__":
    # Input optimisation method details
    method_type = 'random'  # input method type: 'defined' for chosen parameter values; 'exhaust' for exhaustive; 'random' for random combination from exhaustive; 'simple' for simple; 'hone' for honing; 'recover' to export last data set
    method_metric = 'user def'  # input metric defition: 'max' for maximising intensity of first chromatogram; 'ratio' for maximising intensity of first chromatogram to second; 'user def' to define own metric (below)
    method_break = 0  # input a factor for which the method will break and move onto the next parameter, if the new metric falls below method_break * previous (simple only).
    n_random_points = 60  # input number of random combinations of parameters to guess, must be >0 (random and hone only).
    n_honing_points = 60  # input number of subsequent honing points required, must be >0 (hone only).

    # Input parameter details, including bounds
    tab_names = ['ES', 'Instrument']  # input names of param_in_tab in form ['Tab 1', ..., 'Tab X'] or None or [] if only tab used.
    tab_rows = []  # input row of each tab in form [1, ..., 2] or None or [] if all param_in_tab on same row. Only two rows are currently supported.
    param_names = ['Capillary', 'Sampling Cone', 'Source Offset', 'Nebuliser', 'TrapCE', 'TransCE']  # input names of parameters in form ['Param 1, ..., Param X']
    default_params = [3, 0, 50, 3, 5, 0]  # input default start values in the form [0, ..., 0].
    param_bounds = [(0, 5, 1), (0, 100, 10), (0, 100, 10), (2.5, 6, 0.5), (0, 10, 1), (0, 10, 1)]  # input parameters bounds (lower bound, upper bound, increment value) in form e.g. [(X1, X2, X3), ..., (Y1, Y2, Y3)].
    param_in_tab = [1, 1, 1, 1, 2, 2]  # enter param_in_tab required for each parameter in form [1, ..., X] or None or [] if all parameters in Tab 1.
    chrom_num = 0  # enter number of chromatograms in use. Input 0 if chromatograms cannot be copied. Else input integer > 0.

    learn_coord = False  # input True if wanting to learn coordinates of required param_in_tab/boxes/chromatograms, else False.

    # Input system times
    param_change_delay = 10  # input delay between change of parameters in seconds
    stabil_delay = 2  # input time taken for system to stabilise in seconds
    hold_end = 60  # input time desired to hold optimal parameters immediately before termination in seconds

    # Input output file/folder locations
    output_file = r'C:\Users\Waters\Documents\OptiMS\MS_opti_output'  # input location of optimisation file output as r'file_location\name' - do not include file extension, e.g. C:\Users\Waters\Documents\OptiMS\MS_opti_output.
    input_file = r''  # input file location of previously saved parameter details (defined only).
    pic_folder = r''  # input folder location for saving pictures as r'folder_location\folder_name' if desired - else put False or r''

    opti_param, opti_store_df = run_optims(param_names, default_params, param_bounds, tab_names=tab_names,
                                           tab_rows=tab_rows, param_in_tab=param_in_tab, method_type=method_type,
                                           method_metric=method_metric, break_fac=method_break, chrom_num=chrom_num,
                                           param_change_delay=param_change_delay, stabil_delay=stabil_delay,
                                           hold_end=hold_end, n_random_points=n_random_points,
                                           n_honing_points=n_honing_points, learn_coord=learn_coord,
                                           output_file=output_file, input_file=input_file, pic_folder=pic_folder)
