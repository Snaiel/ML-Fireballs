
from __future__ import absolute_import, division, print_function

#import modules
import os
import argparse
import itertools

# import science modules
from astropy.table import Table, Column, QTable
from astropy.time import Time, TimeDelta
from astropy import units as u
import numpy as np


from astropy.utils.data import download_file
from astropy.utils import iers
iers.IERS.iers_table = iers.IERS_A.open(download_file(iers.IERS_A_URL, cache=True))

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA



def computeVelocity(table, xcolname, tcolname='datetime', order=1, extrapol_edge=True):
    
    
    n = len(table)
    
    t = Time(table[tcolname].data.tolist()) 
    #relative_time = Time(t - t[0], format='sec').sec

    delta_X = np.ones(n)
    delta_T = np.ones(n)

    err_plus_dX_dT = np.ones(n) * np.nan
    err_minus_dX_dT = np.ones(n) * np.nan

    # Compute deltas in time and position
    # Order 1: indices [1,n] are comptuted
    # Order 2: indices [1,n-1] are comptuted, using a second order kangaroo method to mix Start with Starts and Ends with Ends
    if order == 1:
        delta_T[1:] = TimeDelta(t[1:] - t[0:n - 1], format='sec').sec
        delta_X[1:] = table[xcolname][1:] - table[xcolname][0:n - 1]
    elif order == 2:
        delta_T[1:n - 1] = TimeDelta(t[2:] - t[0:n - 2], format='sec').sec
        delta_X[1:n - 1] = table[xcolname][2:] - table[xcolname][0:n - 2]
    else:
        pass
    # delta_T=Time(t[order:n]-t[0:n-order],format='sec').sec

    # Compute velocities
    vel_X = delta_X / delta_T

    if order == 1:
        vel_X[0] = np.nan
    elif order == 2:
        vel_X[0] = np.nan
        vel_X[-1] = np.nan
    else:
        pass

    # Return the new velocity column
    theUnit = table[xcolname].unit / u.second
    return vel_X * theUnit, err_plus_dX_dT * theUnit, err_minus_dX_dT * theUnit



def compute_norm(PP_table, PP_file=''):
    #typicalMaxTimingError = 1 * u.second
    #overallNumPoints = str(len(PP_table))
    # get errors, and only consider subset of points with solid timing (error < 1 sec for example)
    #PP_table = PP_table[(PP_table['time_err_plus'] < typicalMaxTimingError) &
    #                    (PP_table['time_err_minus'] < typicalMaxTimingError)]
    #print(("# points good timing / # points overall: " + str(len(PP_table)) + " / " + overallNumPoints))

    dxdt, errplusdxdt, errminusdxdt = computeVelocity(PP_table, 'x_image', tcolname='datetime', order=1, extrapol_edge=False)
    dydt, errplusdydt, errminusdydt = computeVelocity(PP_table, 'y_image', tcolname='datetime', order=1, extrapol_edge=False)

    col_dxdt = Column(name="dxdt_pixel", data=dxdt)
    col_dydt = Column(name="dydt_pixel", data=dxdt)

    PP_table.add_column(col_dxdt)
    PP_table.add_column(col_dydt)

    ddt = Column(name="ddt_pixel", data=LA.norm((dxdt, dydt), axis=0) * u.pixel / u.second)
    PP_table.add_column(ddt)
    return PP_table


def plot_norm(PP_table, fireballName=''):
    if fireballName == '':
        try:
            fireballName = PP_table.meta['event_codename']
        except:
            fireballName = 'fireball'
            pass
    try:
        telescope = PP_table.meta['telescope']
    except:
        telescope = ''
        pass
    try:
        location = PP_table.meta['location']
    except:
        location = ''
        pass
    table = QTable(PP_table)
    t = Time(table['datetime'].data.tolist())
    #numPoints = len(t)
    palette = itertools.cycle(sns.color_palette())
    color = next(palette)
    plt.plot_date(t.plot_date, PP_table['ddt_pixel'],
                  xdate=True, ydate=False,
                  color=color, ls='--', ms=10.0)
    plt.ylim(ymin=0)

    plt.xlabel("Time (t0 + ). t0= " + str(Time(t[0], format='isot', out_subfmt='date')) + " UTC")
    plt.ylabel("Speed (pixel/s)")
    plt.title(fireballName + " " + telescope + " " + location)
    plt.grid(False)
    plt.draw()
    #plt.gcf().autofmt_xdate()  # orient date labels at a slant


def show_norm(PP_table):
    PP_table = compute_norm(PP_table)
    plot_norm(PP_table)
    plt.show()


def save_norm_plot(PP_table, PP_file=''):
    PP_table = compute_norm(PP_table)
    plot_norm(PP_table)
    dirname = os.path.dirname(PP_file)
    basename = os.path.basename(PP_file).split('.')[0]
    fname = os.path.join(dirname, basename + "_" + "dpix" + "_dT_vs_t" + ".png")
    plt.savefig(fname)


def main():
    parser = argparse.ArgumentParser(description='Check meteor point picking velocity in pixel coordinates')
    parser.add_argument("-d", "--datafile", type=str, help="input point picking file")
    args = parser.parse_args()

    PP_file = args.datafile

    PP_table = Table.read(PP_file,format='ascii.ecsv',guess=False, delimiter=',')
    
    show_norm(PP_table)

    # save_norm_plot(PP_table,PP_file=PP_file)



if __name__ == "__main__":
    main()
