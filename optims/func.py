"""General OptiMS functions"""

import matplotlib.pyplot as plt
import pandas as pd
import pyautogui as pg
import openpyxl
import os.path
import string
import sys
import time
import traceback
pg.FAILSAFE = True


# Print text
def gen_err(text):
    print(traceback.format_exc())
    print(text)
    sys.exit(1)


# Single left click at specified coordinates
def left_click(coord):
    pg.click(coord[0], coord[1], clicks=1, button='left')


# Double left click at specified coordinates
def left_click_double(coord):
    pg.click(coord[0], coord[1], clicks=2, button='left')


# Enter number
def enter_num(i):
    pg.typewrite(str(i))


# Double left click and enter parameter value
def left_click_enter_param(coord, i):
    left_click_double(coord)
    pg.typewrite(['backspace'])
    pg.typewrite(['backspace'])
    pg.typewrite(str(i))
    pg.typewrite(['enter'])


# Get coordinates
def coord_find(i):
    coord_find_explain_text = 'Hover mouse over ' + i + ' and press enter'
    print(coord_find_explain_text)
    input()
    return pg.position()


# Print coordinates
def coord_pos_find():
    coord_find_explain_text = 'Hover mouse over coordinate and press enter for next coordinate, or type any letter and press enter to exit'
    print(coord_find_explain_text)
    all_coord = []
    last_input = ''
    while last_input == '':
        last_input = input()
        print(pg.position())
        all_coord.append(pg.position())
    print(all_coord)


# Get location of image
def get_pic_loc(pic_name, alt_text, confidence=0.7):
    if os.path.exists(pic_name) and pg.locateCenterOnScreen(pic_name, confidence=confidence) is not None:  # find the Chromatogram.png image on the desktop.
        coord = pg.locateCenterOnScreen(pic_name, confidence=confidence)
    else:
        coord = coord_find(alt_text)
    return coord


# Copy selected ion chromatogram
def copy_sic(param, chrom_coord, clicks1, link_key, clicks2, time_delay):
    for i in clicks1:
        left_click((chrom_coord[0] + i[0], chrom_coord[1] + i[1]))  # Click on the 'Display' button.
    if len(param) > 0:
        enter_num(param[0])
    if len(param) > 1:
        pg.typewrite(link_key) # pg.hotkey('tab')  # Press enter to tab to next option
        enter_num(param[1])
    pg.typewrite(['enter'])  # Press enter to load species chromatogram
    time.sleep(time_delay)  # Activates time lag in case MassLynx is slow, eg due to windows animations.
    for i in clicks2:
        left_click((chrom_coord[0] + i[0], chrom_coord[1] + i[1]))  # Click on the 'Copy' button.


# Average selected ion chromatogram
def avg_sic(chrom_coord, spec_coord, range_coord, clicks2, time_delay=5):
    pg.moveTo((chrom_coord[0] + range_coord[0][0], chrom_coord[1] + range_coord[0][1]))
    pg.dragTo((chrom_coord[0] + range_coord[1][0], chrom_coord[1] + range_coord[1][1]), button='right')  # Right click and drag to average spectrum
    combine_coord = pg.locateCenterOnScreen('Combine.png', confidence=0.7)
    if combine_coord is None:
        time.sleep(time_delay)
    else:
        while combine_coord is not None:
            combine_coord = pg.locateCenterOnScreen('Combine.png', confidence=0.7)
    for i in clicks2:
        left_click((spec_coord[0] + i[0], spec_coord[1] + i[1]))  # Click on the 'Copy' button.
        pg.typewrite(['delete', 'enter'])


# Get data ranges
def get_range(i, stabil_delay=None, scan_time=None, scan_num=None, hold_end=None):
    if isinstance(i, list):
        ranges = i
    elif 'output_file' in i or isinstance(i, str):
        if 'output_file' in i:
            df = pd.read_excel(i + '_optims_output.xlsx', sheet_name='Output')
        else:
            df = pd.read_excel(i, sheet_name='Output')

        if hold_end != 0:
            scan_num = [scan_num] * len(df.iloc[:, 0].tolist())
        else:
            scan_num = [*[scan_num] * (len(df.iloc[:, 0].tolist()) - 1), hold_end]
        ranges = [(round(i + (stabil_delay / 60), 3), round(i + ((stabil_delay + scan_num * scan_time) / 60), 3))
                  for i in df.iloc[:, 0].tolist()]
    return ranges


# Plot extracted data
def plot_data(df, x_header, y_headers, x_label, y_label, output_file, legend=True):  # Plots the Normalized dataset
    fig = df.plot(x=x_header, y=y_headers, figsize=(6.5, 5), linewidth=1, kind='line',
                  legend=True, fontsize=9, cmap=plt.cm.tab10)
    # fig.set_title(exp_name, fontdict={'fontsize': 12, 'color': 'k'})
    fig.legend(loc='best', frameon=False, fontsize=9).set_visible(legend)
    fig.set_xlabel(x_label, fontdict={'fontsize': 10})  # Fontdict can also include; 'family': 'Arial',  'color': 'r', 'weight': 'bold', 'fontsize':10, 'style': 'italic', etc
    fig.set_ylabel(y_label, fontdict={'fontsize': 10})
    fig.spines['right'].set_visible(False)  # Removing the spines top and right
    fig.spines['top'].set_visible(False)
    # plt.show()
    plt.savefig(output_file + '.png', dpi=300)  # Save the figure, dpi specifies size.
    return


# Export to Excel
def export_excel(df, output_file, sheet_names, mode='w', if_sheet_exists=None):
    data_store_try = False
    while data_store_try is False:
        try:
            writer = pd.ExcelWriter(output_file, engine='openpyxl', mode=mode, if_sheet_exists=if_sheet_exists)
            for i in range(len(df)):
                df[i].to_excel(writer, sheet_name=sheet_names[i], index=None)
            writer.save()
            data_store_try = True
        except OSError:
            output_file = input('Error! Cannot save file into a non-existent folder. '
                                'Input correct file location (do not use r\'\'). \n')
        except PermissionError:
            input('Error! Export file open. Close and then press enter.')
        except:
            gen_err('Unknown error! Check inputs are formatted correctly. Else examine error messages and review code.')


# Add image to Excel
def add_excel_img(output_file, img_file, sheet_name, pic_stag_num):
    img = openpyxl.drawing.image.Image(img_file + '.png')  # Opens previously made norm plot, to be inserted in Excel.
    wb = openpyxl.load_workbook(output_file + '.xlsx', data_only=True)
    wb[sheet_name].add_image(img, string.ascii_uppercase[pic_stag_num] + '1')  # convert the length of species +5 into corresponding letter, and input image at that column letter and row 1.
    wb.save(output_file + '.xlsx')  # Save the workbook with image.


if __name__ == "__main__":
    coord_pos_find()  # Get mouse coordinates
