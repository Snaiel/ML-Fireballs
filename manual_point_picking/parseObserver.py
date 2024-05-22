#!/usr/bin/env python
# Parse fits header to config file
#
# Version history:
# 1.0	Initial commit Hadrien D
# 1.1	Slight clean, objectified a bit Hadrien D

from __future__ import absolute_import, division, print_function


__author__ = "Hadrien Devillepoix"
__copyright__ = "Copyright 2014, Desert Fireball Network"
__license__ = ""
__version__ = "1.1"
__scriptName__ = "parseObserver.py"


import os
import argparse
import configparser
import sys

import numpy as np
import astropy.io.fits as pyfits

from astropy.time import Time
from astropy.coordinates import Longitude
from astropy.time.core import TimeDelta

from astropy.utils.iers import IERSRangeError

import dfn_utils


SENSOR_SIZE_CATALOG = dfn_utils.SENSOR_SIZE_CATALOG.items()

FISHEYE_LENS_PROJECTION_CATALOG = dfn_utils.FISHEYE_LENS_PROJECTION_CATALOG



class UnknownHardwareError(Exception):

    def __init__(self, value, hw_type='Undefined hardware'):
        self.value = value
        self.hw_type = hw_type

    def __str__(self):
        return repr(self.value + ' is not a recognised type of ' + self.hw_type)
    


def fov(focal_length, frame_size, lenstype, lens='unknown'):
    '''
    Calculates the infinity field of view depending on the type of lens
    focal_length: focal length
    frame_size: sensor size (same unit as focal_length)
    lenstype: usually fisheye or rectilinear
    lens: lens model
    '''
    if lenstype.lower() == 'rectilinear':
        return np.rad2deg(2 * np.arctan(1.0 * frame_size/(focal_length * 2)))
    elif lenstype.lower() == 'fisheye':
        try:
            projection = FISHEYE_LENS_PROJECTION_CATALOG[lens.lower()]
        except:
            raise UnknownHardwareError(lenstype, 'lens model wrt projection')
        if projection == 'stereographic':
            return np.rad2deg(4 * np.arctan(1.0 * frame_size/(focal_length * 4)))
        elif projection == 'equisolid':
            return np.rad2deg(4 * np.arcsin (1.0 * frame_size/(focal_length * 4)))
        elif projection == 'equidistance':
            return np.rad2deg(1.0 * frame_size/focal_length)
        elif projection == 'orthogonal':
            return np.rad2deg(2 * np.arcsin(1.0 * frame_size/(focal_length * 2)))
        else:
            raise KeyError(lenstype, 'unknown lens projection')
    else:
        raise KeyError(lenstype, 'unknown lens')
    
    



class dfn_image_metadata:

    def __init__(self, prihdr, overridegps=False):
        self.naxis1 = prihdr['NAXIS1']
        self.naxis2 = prihdr['NAXIS2']
        self.isodate_start_obs = prihdr['DATE-OBS']
        #nonisodate_start_obs = prihdr['OBSTIME']
        self.expoDurSec = float(prihdr['EXPTIME'])

        self.locinfopresent = str(prihdr['LOCINFO'])
        if self.locinfopresent == "YES":
            self.longitude = float(prihdr['SITELONG'])
            self.latitude = float(prihdr['SITELAT'])
            self.altitude = float(prihdr['SITEALT'])
            self.dfn_station_codename = str(prihdr['SITEDFN'])
            self.location = str(prihdr['SITELOC'])
            try:
                self.gps_lock = str(prihdr['GPSLOCK'])
            except:
                self.gps_lock = "MAYBE"
                pass

        if self.locinfopresent != "YES":
            sys.stderr.write(dfn_utils.fail + " ERROR: Location information is not present" + "\n")
            sys.stderr.write("LOCINFO: " + self.locinfopresent + "\n")
            exit(1)

        if self.gps_lock != "Y":
            if overridegps == True:
                print(dfn_utils.warning + " Overriding GPS lock... Location information is possibly unreliable (no GPS lock in the data)" + "\n")
            else:
                sys.stderr.write(dfn_utils.fail + " Location information is possibly unreliable (no GPS lock in the data" + "\n")
                sys.stderr.write("gps_lock: " + self.gps_lock + "\n")
                sys.stderr.write("LOCINFO: " + self.locinfopresent + "\n")
                exit(1)

        # use iso start obs as timing for image
        self.time_start_obs = Time(self.isodate_start_obs, format='isot')
        # then add half of exposure duration to get mid obs time
        self.halfExpo = TimeDelta((self.expoDurSec / 2.0), format='sec')
        self.time_mid_obs = self.time_start_obs + self.halfExpo
        # correct for UTC-UT1 drift
        try:
            self.time_start_obs.delta_ut1_utc = self.time_start_obs.delta_ut1_utc
            self.time_mid_obs.delta_ut1_utc = self.time_mid_obs.delta_ut1_utc
        except IERSRangeError as e:
            print('WARNING: {}'.format(e))
            self.time_start_obs.delta_ut1_utc = 0.
            self.time_mid_obs.delta_ut1_utc = 0.
            

        # compute sidereal time
        self.longitude_Object = Longitude(self.longitude, 'degree')
        self.sidereal_time = self.time_mid_obs.sidereal_time(kind='apparent', longitude=self.longitude_Object, model=None)

        try:
            self.tracking = str(prihdr['TRACKIN'])
        except:
            self.tracking = 'NO'
            
        try:
            self.original_raw_filename = str(prihdr['FILENAME'])
        except:
            self.original_raw_filename = 'unknown'

            # BITPIX  =                   16 / array data type
        try:
            self.creator = str(prihdr['CREATOR'])
        except:
            self.creator = "Unknown"
        try:
            self.lens = str(prihdr['LENS'])
        except:
            self.lens = "Unknown"
        try:
            self.isospeed = str(prihdr['ISOSPEED'])
        except:
            self.isospeed = "N/A"
        self.lenstype = str(prihdr['LENSTYPE'])
        
        if self.lenstype == 'fisheye':
            self.obs_az = 0.0
            self.obs_ev = 90.0
            self.fov_horiz = 180.0
            self.fov_vert = 180.0
        
        self.instrument = str(prihdr['INSTRUME'])
        try:
            self.camera = str(prihdr['CAMERA'])
            if 'SX Superstar' in self.camera:
                self.instrument = 'SX Superstar allsky'
        except:
            pass
        try:
            self.cropped = str(prihdr['CROPPED']).lower() in ("true")
        except:
            self.cropped = False
        try:
            self.origin = str(prihdr['ORIGIN'])
        except:
            self.origin = "Unknown"
        try:
            self.cam = str(prihdr['TELESCOP'])
        except:
            self.cam = str(prihdr['SITEDFN'])
        self.filterrpat = str(prihdr['FILT_PAT'])
        self.focal = float(prihdr['FOCAL'])
        self.aperture = float(prihdr['APERTUR'])
        self.exptime = float(prihdr['EXPTIME'])
        try:
            self.observer = str(prihdr['OBSERVER'])
        except:
            self.observer = "Unknown"
        try:
            self.filterprocess = str(prihdr['COLORTYP'])
        except:
            try:
                self.filterprocess = str(prihdr['FILTER'])
            except:
                self.filterprocess = "Unknown"
        try:
            self.filtercam = str(prihdr['FILT_PAT'])
        except:
            self.filtercam = "Unknown"
        try:
            self.binning = str(prihdr['XBINNING']) + "X" + str(prihdr['YBINNING'])
        except:
            self.binning = "NONE"
        if self.cropped:
            self.crop_x0 = int(prihdr['CROPX0'])
            self.crop_x1 = int(prihdr['CROPX1'])
            self.crop_y0 = int(prihdr['CROPY0'])
            self.crop_y1 = int(prihdr['CROPY1'])
            self.original_naxis_1 = int(prihdr['ONAXIS1'])
            self.original_naxis_2 = int(prihdr['ONAXIS2'])
            self.crop_factor = float(prihdr['ONAXIS2'])/float(prihdr['NAXIS2'])
        else:
            self.crop_factor = 1.0
            
            
    def pixscale_from_header(self):
        '''
        Calculate the pixel size (in arcseconds)
        header: image FITS header
        '''
        # read sensor size
        try:
            UPPER_SENSOR_SIZE_CATALOG = { k.upper() : v for k, v in dfn_utils.SENSOR_SIZE_CATALOG.items() }
            short_side_length = UPPER_SENSOR_SIZE_CATALOG[self.instrument.upper()][1]
            short_side_full_pixcount = UPPER_SENSOR_SIZE_CATALOG[self.instrument.upper()][3]
        except:
            raise UnknownHardwareError(self.lenstype, 'camera/sensor')
        
        # read binning applied
        try:
            binning = int(self.binning.split('X')[0])
        except:
            binning = 1
            
        theoritical_fov = fov(float(self.focal),
                              short_side_length,
                              self.lenstype,
                              lens=self.lens)
        return 1.0 * theoritical_fov / (short_side_full_pixcount / binning) * 3600.0
    
            
    def fov_estimator(self):
        '''
        Estimates the FoV of an image
        '''
        
        pixscale = self.pixscale_from_header()
        
        # use naxis2 = small side
        return self.naxis2 * pixscale / 3600.0

        
    
    def toDict(self):
        dico = {'isodate_start_obs': self.isodate_start_obs,
                'isodate_mid_obs': self.time_mid_obs.isot,
                'jd_start_obs': self.time_start_obs.jd,
                'jd_mid_obs': self.time_mid_obs.jd,
                #'sidereal_time_hms': str(self.sidereal_time),
                #'sidereal_time_hour': self.sidereal_time.hour,
                'obs_latitude': self.latitude,
                'obs_longitude': self.longitude,
                'obs_elevation': self.altitude,
                'dfn_camera_codename': self.dfn_station_codename,
                'location': self.location,
                'location_info': self.locinfopresent,
                'gps_lock': self.gps_lock,
                'tracking': self.tracking,
                'NAXIS1': self.naxis1,
                'NAXIS2': self.naxis2,
                'cx': self.naxis1,
                'cy': self.naxis2,
                'creator': self.creator,
                'telescope': self.cam,
                'iso': self.isospeed,
                'lens_type': self.lenstype,
                'instrument': self.instrument,
                'lens': self.lens,
                'cropped': self.cropped,
                'origin': self.origin,
                'filter_pattern': self.filterrpat,
                'focal': self.focal,
                'aperture': self.aperture,
                'exposure_time': self.exptime,
                'observer': self.observer,
                'processing_filter': self.filterprocess,
                'camera_filter': self.filtercam,
                'original_raw_filename' : self.original_raw_filename}
        
        if self.cropped:
            dico['crop_x0'] = self.crop_x0
            dico['crop_x1'] = self.crop_x1
            dico['crop_y0'] = self.crop_y0 
            dico['crop_y1'] = self.crop_y1
            dico['original_naxis_1'] = self.original_naxis_1
            dico['original_naxis_2'] = self.original_naxis_2
            
        if hasattr(self, 'obs_az') and self.obs_az:
            dico['obs_az'] = obs_az
            dico['obs_ev'] = obs_ev
            dico['fov_horiz'] = fov_horiz
            dico['fov_vert'] = fov_vert
        
            
        for k in dico:
            if isinstance(dico[k], np.generic):
                #dico[k] = np.asscalar(dico[k])
                dico[k] = dico[k].item()

        return dico
    
    
def read_fits_header(inputfile):
    '''
    read FITS header
    '''
    
    hdulist = pyfits.open(inputfile)

    hdulist[0].verify('silentfix')

    prihdr = hdulist[0].header
    
    return prihdr


def get_pixscale(image_file):
    header = read_fits_header(image_file)
    dim = dfn_image_metadata(header, overridegps=True)
    return dim.pixscale_from_header()

def get_fov_size(image_file):
    header = read_fits_header(image_file)
    dim = dfn_image_metadata(header, overridegps=True)
    return dim.fov_estimator()


class ThePixscaleException(Exception):
    
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def test_pixscale_similarity(image_file,pixscale_true):
    pixscale_theo = get_pixscale(image_file)
    percent_change = 100. * abs(pixscale_true - pixscale_theo) / pixscale_theo
    print('Theoritical pixscale (arcsec): ' + str(pixscale_theo))
    print('Pixscale calculated through astrometry (arcsec): ' + str(pixscale_true))
    print('Pixscale variation (%): ' + str(percent_change))
    if percent_change > 20.:
        raise ThePixscaleException("Plate solver must be wrong")


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Parse fits header')
    inputgroup = parser.add_mutually_exclusive_group(required=True)
    inputgroup.add_argument("-i", "--inputfile", type=str, help="input image file name")
    inputgroup.add_argument("-d", "--inputdirectory", type=str, help="input directory for images with extension " + dfn_utils.rawExtensionDefault)
    parser.add_argument("-O", "--overridegps", action="store_true", default=False, help="Will raise warning instead of error if GPS lock is not present")
    parser.add_argument("-w", "--nooverwriteexisting", action="store_true", default=False, help="Will not overwrite observer file if already present")

    args = parser.parse_args()

    dirList = []

    if args.inputfile != None:
        dirList.append(args.inputfile)
        dirName = os.path.dirname(args.inputfile)
    elif os.path.isdir(args.inputdirectory):
        directory = args.inputdirectory
        # list all fits files in this directory
        dirList = dfn_utils.resolve_glob(dfn_utils.fitsExtension, directory=directory, prefix="")
        dirName = directory
    else:
        exit(1)

    if not os.access(dirName, os.W_OK):
        print(dfn_utils.fail + __scriptName__ + " does NOT have write permissions in " + dirName)
        exit(1)

    numberoffiles = len(dirList)
    currentfile = 0

    for i in dirList:
        currentfile += 1
        print("\033[1;34mProcessing image " + str(currentfile) + "/" + str(numberoffiles) + " : " + i + "\033[0m")

        inputfile = i
        # outputascii=args.outputfile
        # generate output file name
        outputfilename = os.path.splitext(inputfile)[0] + "_observer.cfg"

        # test existence
        if os.path.isfile(outputfilename):
            if args.nooverwriteexisting:
                print(dfn_utils.info + " Observer file " + outputfilename + " already present. Skipping " + i)
                continue
            else:
                print(dfn_utils.warning + " Overwriting " + outputfilename)


        prihdr = read_fits_header(inputfile)

        dfnmeta = dfn_image_metadata(prihdr, overridegps=True)

        # write whatever needs to be written in the Observer config file
        config = configparser.RawConfigParser()

        firstsection = 'Observer'
        config.add_section(firstsection)

        config.set(firstsection, 'isodate_start_obs', dfnmeta.isodate_start_obs)
        config.set(firstsection, 'isodate_mid_obs', dfnmeta.time_mid_obs.isot)
        config.set(firstsection, 'jd_start_obs', dfnmeta.time_start_obs.jd)
        config.set(firstsection, 'jd_mid_obs', dfnmeta.time_mid_obs.jd)
        config.set(firstsection, 'sidereal_time_hms', dfnmeta.sidereal_time)
        config.set(firstsection, 'sidereal_time_hour', dfnmeta.sidereal_time.hour)
        config.set(firstsection, 'latitude', dfnmeta.latitude)
        config.set(firstsection, 'longitude', dfnmeta.longitude)
        config.set(firstsection, 'altitude', dfnmeta.altitude)
        config.set(firstsection, 'dfn_station_codename', dfnmeta.dfn_station_codename)
        config.set(firstsection, 'location', dfnmeta.location)
        config.set(firstsection, 'location_info', "YES")
        config.set(firstsection, 'gps_lock', "Y")
        config.set(firstsection, 'tracking', dfnmeta.tracking)

        secondsection = 'Camera'
        config.add_section(secondsection)
        config.set(secondsection, 'NAXIS1', dfnmeta.naxis1)
        config.set(secondsection, 'NAXIS2', dfnmeta.naxis2)

        with open(outputfilename, 'w') as configfile:
            config.write(configfile)

        print(dfn_utils.ok + " Observer parameters written to config file: " + configfile.name)


if __name__ == '__main__':
    main()
