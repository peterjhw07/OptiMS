"""Mass spectra from averaging chromatogram program - designed for use with Waters MassLynx"""

import pandas as pd
import pyautogui as pg
import master_grab as mg


def mz_grab(ranges, scan_time, param_change_delay, stabil_delay, names=None, time_delay=0):
    """
    Grabs chromatograms of defined m/z values or ranges

    Params
    ------
    ranges : list of tuple or float
        Default (or starting) values of param_names
    time_delay : int
        Time delay used if software running slow. Default is 0

    Returns
    -------
    all_spec : list of pandas.DataFrame
        List of mz spectra DataFrames
    """

    clicks1 = [(60, 30), (60, 240), (280, 240)]  # Clicks required to move around interface.

    ranges = mg.get_range(ranges, param_change_delay, stabil_delay)  # Get chrom_ranges for averaging if specified as file instead of list.

    if names is None:
        names = ['chrom_' + str(i[0]) for i in ranges]

    chrom_coord = mg.get_pic_loc('Chromatogram.png', '\'Chromatogram\' Chromatogram window title') # Get location of spectrum and chromatogram windows
    spec_coord = mg.get_pic_loc('Spectrum.png', '\'Spectrum\' of Spectrum window title')
    range_coord = [(-18, 200), (pg.size()[0], 200)]  # Get coordinates for averaging limits

    all_spec = []
    for i in range(len(ranges)):
        # mg.copy_sic(ranges[i], chrom_coord, clicks1, 'tab', [(70, 50)], time_delay)  # Input chromatogram bounds
        # mg.avg_sic(chrom_coord, spec_coord, range_coord, [(90, 50)])  # Get average of chromatogram region and copy spec
        spec = pd.read_clipboard(header=None)
        spec.columns=['m/z', names[i]]  # Make DataFrame from spec
        spec.loc[:, names[i]].div(round((abs(ranges[i][1] - ranges[i][0])) / (scan_time / 60), 0))  # Convert sum intensity to average intensity
        all_spec.append(spec)
    return all_spec


def mz_grab_process(df, output_file, get_csv=False, get_excel=False, get_pic=False):
    """
    Processes mz_grab outputted DataFrames as specified

    Params
    ------
    df : pandas.DataFrame
        DataFrame of mz spectra for the defined ranges
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
        for i in range(len(df)):
            df[i].to_csv(output_file + df[i].columns[1] + '.txt', index=None, sep=',', mode='w')
    if get_excel is True:
        mg.save_excel(df, names, output_file)
    if get_pic is True:
        for i in range(len(df)):
            mg.plot_data(df[i], df[i].columns[0], df[i].columns[1], 'm/z', 'Intensity',
                         output_file, df[i].columns[1], legend=False)
    if get_excel is True and get_pic is True:
        for i in range(len(df)):
            mg.add_excel_img(output_file, df[i].columns[1], df[i].columns[1], len(df[i].columns))


if __name__ == '__main__':
    output_file = r'C:\Users\Peter\Documents\Postdoctorate\Programs\chrom_grab_test'  # Input the directory to save processed data, eg: output_dir = r'C:\Users\IanC\Documents\Experiments'
    ranges = r'C:\Users\Peter\Documents\Postdoctorate\Work\CBD to THC\Tandem MS\PJHW23022308_D9-THC_50UM_D8-THC_50UM_TRAPCE_0-30V_MS_opti_output.xlsx'  # Input list of tuples for ranges for averaging or OptiMS output filename in form r'filename'
    scan_time = 1  # Experimental scan time
    param_change_delay = 10  # Insert experimental delay between changing of parameters (only required if using OptiMS to determine ranges)
    stabil_delay = 2  # Insert experimental delay for stabilisation (only required if using OptiMS to determine ranges)
    time_delay = 0  # Insert a time delay between operations if MassLynx is running slow in format: int(x.x). 1.0 should be enough.

    names = [str(i) + 'V' for i in range(32)]

    data = mz_grab(ranges, scan_time, param_change_delay, stabil_delay, names=names, time_delay=time_delay)
    mz_grab_process(data, output_file, get_csv=True, get_excel=True, get_pic=True)
