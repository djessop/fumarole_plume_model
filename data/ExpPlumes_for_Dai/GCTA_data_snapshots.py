#!/usr/bin/env python3
"""
GCTA_data_snapshots.py
"""

from bentPlumeAnalyser import *
from scipy.io.matlab import loadmat
from matplotlib.lines import Line2D

import numpy as np
import matplotlib.pyplot as plt


scale_factor = 38.               # Conversion factor/[pix/cm]


if __name__ == '__main__':
    plt.close('all')
    cols, rows = 7, 4
    fig, axes = plt.subplots(ncols=cols, nrows=rows, 
                             sharex=True, sharey=True, 
                             figsize=(14, 8))

    axes = axes.ravel()

    ind = 0

    for exptNo in range(1, 33):
        root = '/home/david/Modelling/fumarolePlumeModel/' \
            + 'data/ExpPlumes_for_Dai/'
        path = root + './exp%02d/' % exptNo
        try:
            (ax, im, data,
             xexp, zexp, extent, Orig) = open_plot_expt_image(path, axes,
                                                              exptNo, ind,
                                                              scale_factor)
            Ox, Oz = Orig
            ind += 1                # Only update axes index if expt exits
        
        except FileNotFoundError: # TypeError
            continue
            #print('file %s doesn\'t seem to exist...skipping!' % fname)

    # Add a colorbar as separate axes
    cbar_ax = fig.add_axes([.92, .15, .025, .7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Intensity')

    # Legend in the final axes
    custom_lines = [Line2D([0], [0], ls='-', c='w', lw=2),
                    Line2D([0], [0], ls='--', c='w', lw=2)]
    legend = axes[-1].legend(custom_lines, ['GCTA', 'Present'])
    legend.get_frame().set_facecolor('#bbbbbb')

    fig.suptitle(r'Carazzo \& Aubry expts')
    fig.savefig('GCTA_expt_snapshots.pdf', bbox_inches=None)
    fig.savefig('GCTA_expt_snapshots.png', bbox_inches=None)

    plt.show()
