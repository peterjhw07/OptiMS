"""Function maps for defining functions depending on the instrument used"""

import pandas as pd
import pyautogui as pg
from optims import func


# Custom commands to refresh chromatograms
def chrom_refresh_custom(other_coord):
    return None


# Custom commands to copy chromatograms
def get_chrom_custom(chrom_coord, other_coord):
    func.left_click(chrom_coord)
    func.left_click(other_coord[0])
    data = pd.read_clipboard(header=None)
    data.columns = ['Time', 'Intensity']
    return data


# Custom commands to stop acquisition
def stop_aq_custom(stop_coord, other_coord):
    func.left_click(stop_coord)
    return None


# Commands to refresh chromatograms in MassLynx
def chrom_refresh_ml(other_coord):
    return None


# Commands to copy chromatograms in MassLynx
def get_chrom_ml(chrom_coord, other_coord):
    func.left_click(chrom_coord)
    func.left_click(other_coord[0])
    data = pd.read_clipboard(header=None)
    # data = pd.DataFrame(pd.read_clipboard(header=None), columns=['Time', 'Intensity'])
    data.columns = ['Time', 'Intensity']
    return data


# Commands to stop acquisition in MassLynx
def stop_aq_ml(stop_coord, other_coord):
    func.left_click(stop_coord)
    pg.typewrite(['enter'])
    return None


# Commands to refresh chromatograms in Xcalibur
def chrom_refresh_xc(other_coord):
    pg.typewrite(['F5'])
    return None


# Commands to copy chromatograms in Xcalibur
def get_chrom_xc(chrom_coord, other_coord):
    pg.click(chrom_coord[0], chrom_coord[1], button='right')
    pg.click(chrom_coord[0] + 10, chrom_coord[1] + 50)
    pg.click(chrom_coord[0] + 20, chrom_coord[1] + 50)
    data = pd.read_clipboard(header=3)
    data.columns = ['Time', 'Intensity']
    return data


# Commands to stop acquisition in Xcalibur
def stop_aq_xc(stop_coord, other_coord):
    func.left_click(stop_coord)
    return None


# Map to find appropriate refresh chromatograms commands
chrom_refresh_func_map = {
    'MassLynx': chrom_refresh_ml,
    'Xcalibur': chrom_refresh_xc,
    'Custom': chrom_refresh_custom,
}

# Map to find appropriate copy chromatograms commands
get_chrom_func_map = {
    'MassLynx': get_chrom_ml,
    'Xcalibur': get_chrom_xc,
    'Custom': get_chrom_custom,
}

# Map to find appropriate stop acquisition commands
stop_aq_func_map = {
    'MassLynx': stop_aq_ml,
    'Xcalibur': stop_aq_xc,
    'Custom': stop_aq_custom,
}
