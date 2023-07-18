"""Mass spectra from averaging chromatogram program - designed for use with Waters MassLynx"""

import pandas as pd
import pyautogui as pg
import warnings
import func
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def mz_grab(ranges, scan_time, param_change_delay=None, stabil_delay=None, names=None, time_delay=0):
    """
    Grabs chromatograms of defined m/z values or ranges

    Params
    ------
    ranges : list of tuple or float
        Default (or starting) values of param_names
    scan_time : int or float
        Number of scans taken per second
    param_change_delay : int or float
        Time between parameter changes in seconds. Default is 10
    stabil_delay : int or float
        Time required for instrument to stabilise, following parameter changes in seconds. Default is 2
    names : list of str
        Names of species. If names is None, names are of the format 'chrom_' + str(range). Default is None
    time_delay : int
        Time delay used if software running slow. Default is 0

    Returns
    -------
    all_spec : list of pandas.DataFrame
        List of mz spectra DataFrames
    """

    clicks1 = [(60, 30), (60, 240), (280, 240)]  # Clicks required to move around interface.

    ranges = func.get_range(ranges, param_change_delay=param_change_delay, stabil_delay=stabil_delay)  # Get chrom_ranges for averaging if specified as file instead of list.

    if not names:
        names = ['chrom_' + str(i[0]) for i in ranges]

    chrom_coord = func.get_pic_loc('Chromatogram.png', '\'Chromatogram\' Chromatogram window title') # Get location of spectrum and chromatogram windows
    spec_coord = func.get_pic_loc('Spectrum.png', '\'Spectrum\' of Spectrum window title')
    range_coord = [(-18, 200), (pg.size()[0], 200)]  # Get coordinates for averaging limits

    all_spec = []
    for i in range(len(ranges)):
        func.copy_sic(ranges[i], chrom_coord, clicks1, ['tab'], [(70, 50)], time_delay)  # Input chromatogram bounds
        # func.avg_sic(chrom_coord, spec_coord, range_coord, [(90, 50)])  # Get average of chromatogram region and copy spec
        spec = pd.read_clipboard(header=None)
        spec.columns=['m/z', names[i]]  # Make DataFrame from spec
        spec.loc[:, names[i]].div(round((abs(ranges[i][1] - ranges[i][0])) / (scan_time / 60), 0))  # Convert sum intensity to average intensity
        all_spec.append(spec)
    return all_spec


def mz_grab_process(data, output_file, get_csv=False, get_excel=False, get_pic=False):
    """
    Processes mz_grab outputted DataFrames as specified

    Params
    ------
    data : list of pandas.DataFrame
        List of DataFrames of mz spectra for the defined ranges
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
        for i in range(len(data)):
            data[i].to_csv(output_file + '_' + data[i].columns[1] + '.txt', index=None, sep=',', mode='w')
    if get_excel is True:
        func.export_excel(data, output_file + '_mz_data.xlsx', [data[i].columns[1] for i in range(len(data))])
    if get_pic is True:
        for i in range(len(data)):
            func.plot_data(data[i], data[i].columns[0], data[i].columns[1], 'm/z', 'Intensity',
                         output_file + '_' + data[i].columns[1], legend=False)
    if get_excel is True and get_pic is True:
        for i in range(len(data)):
            func.add_excel_img(output_file + '_mz_data', output_file + '_' + data[i].columns[1], data[i].columns[1], len(data[i].columns))


if __name__ == '__main__':
    output_file = r'C:\Users\Peter\Documents\Postdoctorate\Programs\mz_grab_test'  # Input the directory to save processed data, eg: output_dir = r'C:\Users\IanC\Documents\Experiments'
    ranges = r'C:\Users\Peter\Documents\Postdoctorate\Work\CBD to THC\Tandem MS\PJHW23022308_D9-THC_50UM_D8-THC_50UM_TRAPCE_0-30V_MS_opti_output.xlsx'  # Input list of tuples for ranges for averaging or OptiMS output filename in form r'filename'
    scan_time = 1  # Experimental scan time
    param_change_delay = 10  # Insert experimental delay between changing of parameters (only required if using OptiMS to determine ranges)
    stabil_delay = 2  # Insert experimental delay for stabilisation (only required if using OptiMS to determine ranges)
    time_delay = 0  # Insert a time delay between operations if MassLynx is running slow in format: int(x.x). 1.0 should be enough.

    names = [str(i) + 'V' for i in range(32)]

    data = mz_grab(ranges, scan_time, param_change_delay, stabil_delay, names=names, time_delay=time_delay)
    mz_grab_process(data, output_file, get_csv=True, get_excel=True, get_pic=True)
