"""Individual species chromatogram collecting program - designed for use with Waters MassLynx"""

####  MassLynx DataGrab 2.9   ####

## Use this program for quick automated copying and exporting of MassLynx single ion chromatogram (sic) data.
## SIC data is normalised to TIC, then plotted onto a single chromatogram plot, and extracted into an Excel file.
## Input your experiment name (exp_name), m/z of species to copy (species), and a file output directory (output_dir) below, then run script!

## ATTN: Rogue automated mouse movements can be cancelled by quickly moving cursor to any of the four corners of the primary monitor.

import pandas as pd
import pyautogui as pg
import warnings
import func
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def chrom_grab(ranges, names=None, time_delay=0):
    """
    Grabs chromatograms of defined m/z values or ranges

    Params
    ------
    ranges : list of tuple or float
        Default (or starting) values of param_names
    names : list of str
        Names of species. If names is None, names are of the format 'm/z ' + str(value/range). Default is None
    time_delay : int
        Time delay used if software running slow. Default is 0

    Returns
    -------
    raw_df : pandas.DataFrame
        DataFrame of raw chromatograms for TIC and defined species
    norm_df : pandas.DataFrame
        DataFrame of chromatograms normalized by the TIC for the defined species
    """

    clicks1 = [(60, 30), (60, 45)]  # Clicks required to move around interface.
    ranges = [i if isinstance(i, (tuple, list)) else tuple([i]) for i in ranges]
    if not names:
        names = ['m/z ' + str(i[0]) + '-' + str(i[1]) if len(i) > 1 else 'm/z ' + str(i[0]) for i in ranges]

    chrom_coord = func.get_pic_loc('Chromatogram.png', '\'Chromatogram\' Chromatogram window title')  # Get location of chromatogram window

    # func.copy_sic([], chrom_coord, [], '_', [(70, 50)], time_delay)
    tic_df = pd.read_clipboard(header=None)
    raw_df = pd.DataFrame(index=range(tic_df.shape[0]), columns=['Time', 'TIC', *names])
    raw_df.loc[:, 'Time'], raw_df.loc[:, 'TIC'] = tic_df.iloc[:, 0], tic_df.iloc[:, 1]
    norm_df = raw_df.loc[:, ['Time', *names]]
    for i in range(len(ranges)):  # Specifying that for every instance in species, run copy_sic, insert_sic and norm_sic
        # func.copy_sic(ranges[i], chrom_coord, clicks1, '_', [(70, 50)], time_delay)  # Select chromatogram region
        pg.typewrite(['delete', 'enter'])
        data = pd.read_clipboard(header=None)  # Make a new temporary dataframe of new species data. header=None so 1st row is copied, and not excluded as a label
        raw_df.loc[:, names[i]] = data.iloc[:, 1]  # Convert sum intensity to average intensity
    norm_df.loc[:, names] = raw_df.loc[:, names].div(raw_df.loc[:, 'TIC'], axis=0)
    return raw_df, norm_df


def chrom_grab_process(raw_df, norm_df, output_file, get_csv=False, get_excel=False, get_pic=False):
    """
    Processes chrom_grab outputted DataFrames as specified

    Params
    ------
    raw_df : pandas.DataFrame
        DataFrame of raw chromatograms for TIC and defined ranges
    norm_df : pandas.DataFrame
        DataFrame of chromatograms normalized by the TIC for the defined species
    output_file : str
        Output filename and location. Do not include file extensions
    get_csv : bool, optional
        Determines whether DataFrames will outputted as csv files. Default is False
    get_excel : bool, optional
        Determines whether DataFrames will outputted as Microsoft Excel file. Default is False
    get_pic : bool, optional
        Determines whether figures will plotted and saved as png files. Default is False

    """
    if get_csv is True:  # Output data as csv files
        raw_df.to_csv(output_file + '_chrom_data_raw' + '.txt', index=None, sep=',', mode='w')
        norm_df.to_csv(output_file + '_chrom_data_norm' + '.txt', index=None, sep=',', mode='w')
    if get_excel is True:
        func.save_excel([raw_df, norm_df], ['raw', 'norm'], output_file + '_chrom_data')
    if get_pic is True:
        func.plot_data(raw_df, raw_df.columns[0], raw_df.columns[1:], 'Time / min', 'Intensity',
                     output_file + '_chrom_plot_raw', legend=True)
        func.plot_data(norm_df, norm_df.columns[0], norm_df.columns[1:], 'Time / min', 'Relative Intensity',
                     output_file+ '_chrom_plot_norm', legend=True)
    if get_excel is True and get_pic is True:
        func.add_excel_img(output_file + '_chrom_data', output_file + '_chrom_plot_raw', 'raw', len(raw_df.columns))
        func.add_excel_img(output_file + '_chrom_data', output_file + '_chrom_plot_norm', 'norm', len(norm_df.columns))


if __name__ == "__main__":
    ranges = [922.8727, 923.8727, 924.8776, 925.8742, 926.8714, 927.8824, 928.8806, 929.8660, 930.8654, 931.8652, 932.8655, 933.8665]
    time_delay = 0  # Insert a time delay between operations if MassLynx is running slow in format
    output_file = r'C:\Users\Peter\Documents\Postdoctorate\Programs\chrom_grab_test'  # Input the directory to save processed data, eg: output_dir = r'C:\Users\IanC\Documents\Experiments'

    raw_df, norm_df = chrom_grab(ranges, time_delay=time_delay)
    chrom_grab_process(raw_df, norm_df, output_file, get_csv=True, get_excel=True, get_pic=True)

# DONE!  © chagunda@uvic.ca
