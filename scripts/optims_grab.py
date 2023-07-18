"""Grab extra optimisation data from existing OptiMS run program - originally designed for use with Waters MassLynx"""

import pandas as pd
import pyautogui as pg
import pickle
import sys
import warnings
import func
import instrument_map
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def all_chrom_grab(ranges, names=None, other_coord_num=0, software=None, learn_coord=True, time_delay=0):
    """
    Grabs chromatograms of defined m/z values or ranges

    Params
    ------
    ranges : list of tuple or float
        Default (or starting) values of param_names
    names : list of str
        Names of species. If names is None, names are of the format 'm/z ' + str(value/range). Default is None
    other_coord_num : int, optional
        Number of other coordinates required for abstracting chromatogram data
    software : str, optional
        Input software. Currently accepted softwares are MassLynx (Waters), Xcalibur (Thermo) and custom (defined under instrument_map).
        Default is None, which is not compatible with simple or hone method_type.
    learn_coord : bool
        Specifies whether on-screen coordinates need to be learnt. Default is True
    time_delay : int
        Time delay used if software running slow. Default is 0

    Returns
    -------
    df : pandas.DataFrame
        DataFrame of recorded times following instrument parameter alterations, instrument parameter alterations,
        and recorded averages, errors and metrics for open chromatograms or defined species
    """

    if isinstance(ranges, int):
        get_chrom = instrument_map.get_chrom_func_map.get(software)

        if 'MassLynx' in software:
            other_coord_num = 1
        elif 'Xcalibur' in software:
            other_coord_num = 0

        if not names:
            names = ['Chrom ' + str(i) for i in range(ranges)]

        coord_store_filename = 'coord_store_file.pkl'

        if learn_coord:
            chrom_coord = [func.coord_find('chromatogram ' + str(i + 1)) for i in range(ranges)]
            if ranges != 0 and 'MassLynx' in software:
                other_coord = [func.coord_find('copy chromatogram button')]
            else:
                other_coord = [func.coord_find('other coordinate ' + str(i + 1)) for i in range(other_coord_num)]
            all_coord = chrom_coord, other_coord
            with open(coord_store_filename, 'wb') as outp:
                pickle.dump(all_coord, outp, pickle.HIGHEST_PROTOCOL)

        with open(coord_store_filename, 'rb') as inp:
            all_coord = pickle.load(inp)
        chrom_coord, other_coord = all_coord

        get_chrom(chrom_coord[0], other_coord)
        dummy_df = pd.read_clipboard(header=None)
        df = pd.DataFrame(index=range(dummy_df.shape[0]), columns=['Time', *names])
        df.loc[:, 'Time'] = dummy_df.iloc[:, 0]
        for i in range(ranges):  # Specifying that for every instance in species, run copy_sic, insert_sic and norm_sic
            get_chrom(chrom_coord[i], other_coord)
            data = pd.read_clipboard(header=None)  # Make a new temporary dataframe of new species data. header=None so 1st row is copied, and not excluded as a label
            df.loc[:, names[i]] = data.iloc[:, 1]  # Convert sum intensity to average intensity
    elif isinstance(ranges, (list, tuple)) and 'MassLynx' in software:
        clicks1 = [(60, 30), (60, 45)]  # Clicks required to move around interface.
        ranges = [i if isinstance(i, (tuple, list)) else tuple([i]) for i in ranges]
        if not names:
            names = ['m/z ' + str(i[0]) + '-' + str(i[1]) if len(i) > 1 else 'm/z ' + str(i[0]) for i in ranges]

        chrom_coord = func.get_pic_loc('Chromatogram.png', '\'Chromatogram\' Chromatogram window title')  # Get location of chromatogram window

        func.copy_sic([], chrom_coord, [], '_', [(70, 50)], time_delay)  # Select chromatogram region - turn off when testing offline
        dummy_df = pd.read_clipboard(header=None)
        df = pd.DataFrame(index=range(dummy_df.shape[0]), columns=['Time', *names])
        df.loc[:, 'Time'] = dummy_df.iloc[:, 0]
        for i in range(len(ranges)):  # Specifying that for every instance in species, run copy_sic, insert_sic and norm_sic
            func.copy_sic(ranges[i], chrom_coord, clicks1, '_', [(70, 50)], time_delay)  # Select chromatogram region - turn off when testing offline
            pg.typewrite(['delete', 'enter'])
            data = pd.read_clipboard(header=None)  # Make a new temporary dataframe of new species data. header=None so 1st row is copied, and not excluded as a label
            df.loc[:, names[i]] = data.iloc[:, 1]  # Convert sum intensity to average intensity
    else:
        print('Values for ranges only valid of software is MassLynx. '
              'Please change software or set ranges to int of number of already visible chromatograms')
        sys.exit
    return df


def get_all_avg_chrom(output_df, chrom_df, param_change_delay, stabil_delay):
        df = output_df
        for i in range(len(df)):
            avg, error, error_perc = [], [], []
            for j in range(1, chrom_df.shape[1]):
                chrom_df_loc = chrom_df.loc[(chrom_df['Time'] >= (df.iloc[i, df.columns.get_loc('Chrom time')] + (stabil_delay / 60))) & (chrom_df['Time'] <= (df.iloc[i, df.columns.get_loc('Chrom time')] + (param_change_delay / 60)))]
                avg.append(chrom_df_loc[chrom_df.columns[j]].mean())
                error.append(chrom_df_loc[chrom_df.columns[j]].std() / (len(chrom_df_loc[chrom_df.columns[j]]) ** 0.5))
                error_perc.append((error[-1] / avg[-1]) * 100)
            df.at[i, 'Chrom averages'], df.at[i, 'Chrom errors'], df.at[i, 'Chrom errors %'] = avg, error, error_perc
        return df


def optims_grab(import_file, ranges, param_change_delay, stabil_delay, import_sheet_name='Output', names=None,
                other_coord_num=0, software=None, learn_coord=True, time_delay=0):
    chrom_df = all_chrom_grab(ranges, names=names, other_coord_num=other_coord_num, software=software,
                              learn_coord=learn_coord, time_delay=time_delay)
    import_df = pd.read_excel(import_file, sheet_name=import_sheet_name)
    output_df = get_all_avg_chrom(import_df, chrom_df, param_change_delay, stabil_delay)
    return output_df


def optims_grab_process(export_df, import_file, import_sheet_name='Output'):
    func.export_excel([export_df], import_file, [import_sheet_name], mode='a', if_sheet_exists='new')


if __name__ == "__main__":
    # Input import file/folder locations
    import_file = r'C:\Users\Peter\Documents\Postdoctorate\Work\CBD to THC\Tandem MS\PJHW23022301_CBD_100UM_TRANSCE_0-30V_MS_opti_output - optims_grab_test.xlsx'  # Input filename of previous OptiMS output file, including extension, eg: import_file = r'C:\Users\IanC\Documents\Experiments\Exp_optims_output.xlsx'

    # Input species for export
    ranges = 2  # Input int number of chromatograms (e.g. 3) or list of int or tuples (e.g. [922.8727, (923.8727, 924.8776)]
    names = []  # Insert desired names for regions as list, e.g. ['Region 1', 'Region 2', ...] or leave empty for automated naming, i.e. [].

    # Input system times
    param_change_delay = 10  # input delay between change of parameters in seconds
    stabil_delay = 2  # input time taken for system to stabilise in seconds
    hold_end = 10  # input time desired to hold optimal parameters immediately before termination in seconds

    # Input output file/folder locations
    software = 'MassLynx'
    other_coord_num = 1
    learn_coord = True

    time_delay = 0  # Insert a time delay between operations if MassLynx is running slow in format

    output_df = optims_grab(import_file, ranges, param_change_delay, stabil_delay,
                            names=names, other_coord_num=other_coord_num, software=software, learn_coord=learn_coord,
                            time_delay=time_delay)
    optims_grab_process(output_df, import_file)
