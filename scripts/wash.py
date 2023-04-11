"""Infinite fluidics washing program - designed for use with Waters MassLynx"""

import pyautogui as pg
import time
import os.path
import master_grab as mg


def run_wash():
    if os.path.exists('Fluidics.png') and pg.locateCenterOnScreen('Fluidics.png', confidence=0.7) is not None:  # find the Chromatogram.png image on the desktop.
        mg.left_click(pg.locateCenterOnScreen('Fluidics.png', confidence=0.7))
    if os.path.exists('Play.png') and pg.locateCenterOnScreen('Play.png', confidence=0.7) is not None:  # find the Chromatogram.png image on the desktop.
        play_stop_coord = pg.locateCenterOnScreen('Play.png', confidence=0.7)
    else:
        play_stop_coord = mg.get_pic_loc('Stop.png', 'play/stop button')

    refill_coord = mg.get_pic_loc('Refill.png', 'refill button')

    for i in range(0, 100):
        mg.left_click(play_stop_coord)
        print('Stop')
        time.sleep(5)
        mg.left_click(refill_coord)
        print('Refilling')
        time.sleep(20)
        mg.left_click(play_stop_coord)
        print('Play')
        time.sleep(240)
        print('Wash ' + str(i + 1) + ' complete')


if __name__ == "__main__":
    run_wash()
