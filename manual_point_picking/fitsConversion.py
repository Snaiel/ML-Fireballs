#!/usr/bin/env python
# Create FITS images from digital camera raw file
# Currently supports only Bayer filter for de-bayering (RGGBRGGBRGGBRGGB)
# depends on:
#     - dcraw
#     - http://www.astromatic.net/software/stiff which can be installed as RPM (optional)
#     - rawpy (optional)
#
# Version history:
# 1.0   DFN dev initial work, Hadrien Devillepoix
# 2.0   Refactor main call
# 2.1   Added optional outputs for file, saves running time and disk space
# 3.0   Completely refactored
# 4.0   Fixed flip problem
# 4.1   Added temp directory option
# 4.2   Added dark frame subtraction support
# 4.3   Added binning support
# 5.0   Ported to python 3
# 5.1   Removed dependance on temporary files, pipe raw file to fits objet directly
# 5.2   Specify output folder
# 5.3   Refactor: isolate main
# 5.4   Added logging
# 5.5   Refactored: more flexible I/O handling
# 6.0   Switched decoding engine to rawpy
# 6.1   Added absolute timing corrections based on uC triggers



__author__ = "Hadrien A.R. Devillepoix"
__copyright__ = "Copyright 2014-2019, Desert Fireball Network"
__license__ = "MIT"
__version__ = "6.1"
__pipeline_task__ = "raw_to_fits"


# general libs
import os
import sys
import re
import logging
import subprocess
import configparser
from configparser import ConfigParser
from copy import deepcopy
import datetime

# science libs
import numpy as np
from astropy.time import Time, TimeDelta
try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits
    
# local libs
try:
    import rawpy
    RAW_HANDLING_LIB = 'RAWPY' # RAWPY/LIBRAW/DCRAW
except ImportError:
    import netPbm
    RAW_HANDLING_LIB = 'DCRAW'


class ConfigFileNotFoundError(FileNotFoundError):
    pass
    
import dfn_utils
import technical_constants_DFN

# check python version
if sys.version_info < (3, 0):
    raise RuntimeError("Need a python interpreter >= 3.0")

# global static variables
STIFF_CONF_TEMPLATE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dfn_stiff.conf')
TEMP_DIR = '/tmp/'


FITS_EXTENSION = technical_constants_DFN.FITS_EXTENSION
TIF_EXTENSION = technical_constants_DFN.TIF_EXTENSION
DEFAULT_RAW_EXTENSION = technical_constants_DFN.DEFAULT_RAW_EXTENSION

# in seconds
DSLR_TRIGGER_DELAY_CATALOG = technical_constants_DFN.DSLR_TRIGGER_DELAY_CATALOG

SATURATION_VALUE_CATALOG = technical_constants_DFN.SATURATION_VALUE_CATALOG


def colorOptionDictionary(option):
    '''
    Defines a dictionary where each key is the requested color, and value is the suffix in the file name
    '''
    
    #logger = logging.getLogger()

    # Monochrome image with raw pixel values (might look like a chessboard pattern)
    raw = ("raw", 'RAW')
    #raw = ("raw", 'extension':'RAW')

    # Green interpolated: Bilinear interpolation of blue and red pixels using the 4 neighbouring green pixels
    gi = ("green_interpolated", 'G')

    # Color subsets stitched by collapsing rows/columns
    g1 = ("green1_collapsed", 'G1')
    g2 = ("green2_collapsed", 'G2')
    b = ("blue_collapsed", 'B')
    r = ("red_collapsed", 'R')

    # Bin 2x2 mixing all 4 color channels (n+2 bits result) (n is original bit depth, eg. Nikon D800 is 14 bits)
    rggb2x2 = ("rggb_binned_2x2", 'RGB2X2')

    # Bin 2x2 mixing the 2 green color channels (n+1 bits result)
    g2x2 = ("green_binned_2x2", 'G2X2')

    if option == "green":
        outputColors = dict([gi])
    if option == "full":
        outputColors = dict([raw, gi, r, b, rggb2x2, g2x2])
    if option == "rawgreen":
        outputColors = dict([raw, gi])
    if option == "red":
        outputColors = dict([r])
    if option == "blue":
        outputColors = dict([b])
    if option == "colors":
        outputColors = dict([gi, r, b, g1, g2])
    if option == "rgb2x2":
        outputColors = dict([rggb2x2])
    if option == "green2x2":
        outputColors = dict([g2x2])

    return outputColors


def convertToTIFF(fitsFile, stiffconf=STIFF_CONF_TEMPLATE):
    """
    Convert a FITS file to TIFF using Stiff (www.astromatic.net/software/stiff)
    parameters:
        - fitsFile: input FITS file
        - stiffconf: STIFF configuration file (stiff -d can help)
    """
    
    logger = logging.getLogger()
    
    otifffile = os.path.splitext(fitsFile)[0] + "." + TIF_EXTENSION
    tmpstiffconf = os.path.join(TEMP_DIR, 'tmp_stiff_' + os.path.basename(fitsFile) + '.conf')
    if not os.path.isfile(stiffconf):
        raise FileNotFoundError('Cannot find STIFF config file template:' + stiffconf)
    f = open(tmpstiffconf, 'w')
    with open(stiffconf, 'r') as fin:
        f.write(fin.read())
    f.write('OUTFILE_NAME           %s       # Name of the output file\n' % (otifffile))
    f.close()
    os.system("stiff -c " + tmpstiffconf + "   " + fitsFile)
    
    # clean up
    os.remove(tmpstiffconf)
    return otifffile


    

def raw2fits(inputfile, crop="",  dark="", flat='', inverseGrayscale=False):
    """
    Create FITS object from raw file (doesn't write)
    parameters:
        - inputfile: input raw file path
        - crop: use this option if you want to crop the image. string like "x0,y0,size_x,size_y"
        - dark: path to pgm file containing frame to subtract
        - inverseGrayscale
    returns:
        - pyfits object
    """
    
    logger = logging.getLogger()

    # Read RAW meta
    try:
        # Getting the EXIF of raw file with dcraw
        p = subprocess.Popen(["dcraw", "-i", "-v", inputfile], stdout=subprocess.PIPE)
        dcraw_exif = p.communicate()[0].decode('utf-8')
        
        # Catching the Shutter Speed
        m = re.search('(?<=Shutter:).*(?=sec)', dcraw_exif)
        shutterstr = m.group(0).strip()
        if "/" in shutterstr:
            shutterplit = m.group(0).strip().split('/')
            shutter = float(shutterplit[0]) / float(shutterplit[1])
        else:
            shutter = float(shutterstr)

        m = re.search('(?<=Aperture: f/).*', dcraw_exif)
        aperture = m.group(0).strip()

        m = re.search('(?<=ISO speed:).*', dcraw_exif)
        iso = m.group(0).strip()

        m = re.search('(?<=Filename:).*', dcraw_exif)
        original_file = m.group(0).strip()

        m = re.search('(?<=Focal length: ).*(?=mm)', dcraw_exif)
        focal = m.group(0).strip()

        m = re.search('(?<=Camera:).*', dcraw_exif)
        camera = m.group(0).strip()
        
        try:
            m = re.search('(?<=Filter pattern:).*', dcraw_exif)
            filter_pattern = m.group(0).strip()
        except AttributeError:
            # .thumb.JPG do not have this in the exif
            filter_pattern = 'RG/GB'
        
        # Catching the true Image Size
        # Image size:  7362 x 4920
        try:
            m = re.search('(?<=Image size: ).*', dcraw_exif)
            dim_list = m.group(0).strip().split('x')
        except AttributeError:
            m = re.search('(?<=Full size: ).*', dcraw_exif)
            dim_list = m.group(0).strip().split('x')
        NAXIS1 = int(dim_list[0])
        NAXIS2 = int(dim_list[1])


        # Catching the Timestamp
        m = re.search('(?<=Timestamp:).*', dcraw_exif)
        date1 = m.group(0).split()
        months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        date = datetime.datetime(int(date1[4]), months[date1[1]], int(date1[2]), int(date1[3].split(':')[0]), int(date1[3].split(':')[1]), int(date1[3].split(':')[2]))

        # To use for timestamp/timezone correction FIXME
        #print('WARNING WARNING WARNING WARNING WARNING WARNING WARNING')
        #logger.critical('This is a very special mode for timing correction, do not use in PRODUCTION')
        #logger.critical("DATETIME: " + str(date))
        #from datetime import timedelta
        #utc_delta = timedelta(days=0,hours=11,minutes=0, seconds=6)
        #logger.critical(utc_delta)
        #logger.critical(date - utc_delta)
        #date = date - utc_delta
        #logger.critical("DATETIME: " + str(date))
        
        # for img 
        #2019-09-01T04:23:40
        #import dateutil.parser
        #date = dateutil.parser.parse('2019-09-01T04:23:40')

        # formating date
        date = '{0:%Y-%m-%d %H:%M:%S}'.format(date)
        
        exposure_recorded_time = Time(date)
        
        # to to match the timestamp to actual trigger time + trigger delay
        try:
            # D810 56ms
            interval_log_file = dfn_utils.find_log_file(os.path.dirname(inputfile), suffix='_log_interval', extension='txt')
            logger.debug('Found interval log file: {}'.format(interval_log_file))
            fw_string = dfn_utils.search_dfn_operation_log(interval_log_file, key='leostick_version', module='interval_control_lin')
            ###    core_1_temperature_list = re.findall('(?:^Core 1:\s+)([\+-]\d+\.\d+)', sensors, re.MULTILINE)
            # 2017-08-23 18:17:51,447, INFO, interval_control_lin, leostick_version, built:Jun  7 2017 10:12:58 note:kit, small and ext compatible, bulb and non-bulb, pulse frequency and pulse width encoding, ext heating and cooling, single exposure and video camera triggering, exposure:5.0/15s, element period:1000ms, short 0 dash length:200ms, long 1 dash length:600ms, target:unifi^Ad
            #print(re.findall('(?:exposure:\d+\.\d+/)(\d+)', fw_string))
            try:
                cadence = int(re.findall('(?:exposure:\d+\.\d+/)(\d+)', fw_string)[0])
                logger.info('Found cadence {} in interval log {}'.format(cadence, os.path.basename(interval_log_file)))
            except Exception as e:
                logger.error('This must be some old firmware string, cannot find cadence')
                raise
        except Exception as e:
            logger.error(e)
            logger.error('Cannot determine image cadence, setting it to 5 seconds')
            # default to 5 seconds (avoid messing up data too much)
            cadence = 5
            
        
        exposure_triggered_time = dfn_utils.round_to_nearest_n_seconds(exposure_recorded_time, cadence)
        
        logger.info('Image timestamp has been rounded off from {} to {}'.format(exposure_recorded_time.isot, exposure_triggered_time.isot))
        
        try:
            UPPER_DSLR_TRIGGER_DELAY_CATALOG = { k.upper() : v for k, v in DSLR_TRIGGER_DELAY_CATALOG.items() } 
            trigger_delay = TimeDelta(UPPER_DSLR_TRIGGER_DELAY_CATALOG[camera.upper()], format='sec')
        except Exception as e:
            logger.error(e)
            logger.error('Cannot find DSLR trigger delay in database, setting it to 0')
            trigger_delay = TimeDelta(0.0, format='sec')
            
        exposure_corrected_time = exposure_triggered_time + trigger_delay
        
        logger.info('Image timestamp has been offset from {} to {}'.format(exposure_triggered_time.isot, exposure_corrected_time.isot))
        
        

    except:
        logger.error('Failed during DCRAW meta read attempt')
        raise


    
    # Dark subtraction if needed
    if dark != "":
        dark_com = ["-K ", dark]
    else:
        dark_com = []
        
    # Cropping if needed
    if crop == "":
        needsCropping = False
        cropprofile = "None"
        crop_command = []
    else:
        needsCropping = True
        # regex for finding 4 numbers separated by commas
        match = re.search("[0-9]+(,[0-9]+){3}", crop)
        # Crop centered presets
        if 'centered_' in crop and crop.split('_')[1].isnumeric():
            crop_size = int(crop.split('_')[1])
            # top left pixel must have even coordinates, otherwise bayer pattern will be shifted
            x0 = int((NAXIS1 - crop_size) / 2)
            y0 = int((NAXIS2 - crop_size) / 2)
            y_size = crop_size
            x_size = crop_size
            cropprofile = crop
        elif match:
            cropstr = match.group().split(',')
            x0 = int(cropstr[0])
            y0 = int(cropstr[1])
            x_size = int(cropstr[2])
            y_size = int(cropstr[3])
            cropprofile = "Custom"
        else:
            needsCropping = False
            cropprofile = "None"
            logger.error('Crop input is not valid. Crop aborted')
            raise KeyError('Invalid crop parameters')

        # top left pixel must have even coordinates, otherwise bayer pattern will be shifted
        if x0 % 2 == 1:
            x0 += 1
        if y0 % 2 == 1:
            y0 += 1

        # check the requested crop falls into the picture bounds
        if (x0 >= NAXIS1) or (y0 >= NAXIS2) or ((x0 + x_size) >= NAXIS1) or ((y0 + y_size) >= NAXIS2) or (x0 < 0) or (y0 < 0):
            needsCropping = False
            cropprofile = "None"
            OOB_msg = 'Crop input does NOT fall into picture bounds. Crop aborted'
            logger.error(OOB_msg)
            raise IndexError(OOB_msg)

        # use netpbm's pamcut to crop
        crop_command = ['pamcut', '-left', str(x0), '-top', str(y0), '-width', str(x_size), '-height', str(y_size)]
    
    if RAW_HANDLING_LIB == 'RAWPY':
        logger.info("Using RAWPY for decoding")
        # .raw_image contains 32 extra columns (on Sony) filled with nonsense
        raw_data = rawpy.imread(inputfile).raw_image_visible
        if dark != "":
            dark_data = pyfits.open(dark)[0].data
            raw_data = (raw_data - dark_data).astype(np.uint16)
        if flat != "":
            flat_data = pyfits.open(flat)[0].data
            raw_data = (raw_data / flat_data * np.mean(flat_data)).astype(np.uint16)
        if needsCropping:
            im_ppm = raw_data[y0:y0+y_size, x0:x0+x_size]
        else:
            im_ppm = raw_data
            
    
    
    elif RAW_HANDLING_LIB == 'JPG':
        logger.critical('This feature still requires LOTS of testing, NOT ready for production')
        #exit(5)
        
        logger.error('I do not know the size of the thumbail vs the size of a raw image for this camera ({})'.format(camera))
        if not camera in technical_constants_DFN.IMAGE_FULL_VS_THUMB_CATALOG:
            logger.error('I do not know the size of the thumbail vs the size of a raw image for this camera ({}). Update technical_constants_DFN.py with this info.'.format(camera))
            raise ValueError('Unsupported camera')
        
        cam_thumb_crop_details = technical_constants_DFN.IMAGE_FULL_VS_THUMB_CATALOG[camera]
        shift_long_ax = int((cam_thumb_crop_details[0]-cam_thumb_crop_details[2])/2)
        shift_short_ax = int((cam_thumb_crop_details[1]-cam_thumb_crop_details[3])/2)
        logger.debug('Long  axis is {} (RAW) vs {} (thmub JPG). {} more pixels, {} pixel shit induced.'.format(cam_thumb_crop_details[0], cam_thumb_crop_details[2],
                                                                                                        (cam_thumb_crop_details[0]-cam_thumb_crop_details[2]),
                                                                                                        shift_long_ax))
        logger.debug('Short axis is {} (RAW) vs {} (thmub JPG). {} more pixels, {} pixel shit induced.'.format(cam_thumb_crop_details[1], cam_thumb_crop_details[3],
                                                                                                        (cam_thumb_crop_details[1]-cam_thumb_crop_details[3]),
                                                                                                        shift_short_ax))
        
        logger.info("Using imageio for decoding")
        # .raw_image contains 32 extra columns (on Sony) filled with nonsense
        import imageio
        green_bandim = imageio.imread(inputfile, pilmode='RGB')
        
        raw_data = np.zeros((cam_thumb_crop_details[1], cam_thumb_crop_details[0]))
        logger.debug('Created an empty array with size {}'.format(np.shape(raw_data)))
        
        if shift_short_ax==0 and shift_long_ax==0:
            raw_data = green_bandim[:,:,1]
        else:
            raw_data[shift_short_ax:-shift_short_ax,shift_long_ax:-shift_long_ax] = green_bandim[:,:,1]
        
        
        if dark != "":
            dark_data = pyfits.open(dark)[0].data
            raw_data = (raw_data - dark_data).astype(np.uint16)
        if flat != "":
            flat_data = pyfits.open(flat)[0].data
            raw_data = (raw_data / flat_data * np.mean(flat_data)).astype(np.uint16)
        if needsCropping:
            im_ppm = raw_data[y0:y0+y_size, x0:x0+x_size]
        else:
            im_ppm = raw_data
            
            
            
    
    elif RAW_HANDLING_LIB == 'DCRAW':
        logger.info("Using DCRAW for decoding")

        # DCRAW extract command
        # -c option send dcraw output to stdout
        base_command = ["dcraw", "-D", "-4", "-t", "0", "-c"]
        
        # Prepare command for converting the RAW to PPM
        the_command = base_command
        # dark subtraction option
        the_command += dark_com
        # target raw file
        the_command += [inputfile]
                    
        # Read RAW data
        try:
            conv = subprocess.Popen(the_command,
                                    stdout=subprocess.PIPE)
            
            # optional pipe to cropping command
            if len(crop_command) >= 1:
                crop = subprocess.Popen(crop_command,
                                        stdin=conv.stdout,
                                        stdout=subprocess.PIPE)
                end_of_pipe = crop.stdout
            else:
                end_of_pipe = conv.stdout
            
            # pipe output into netPbm object
            im_ppm = netPbm.NetpbmFile(end_of_pipe).asarray()
            
        except Exception as e:
            logger.critical('Failed during DCRAW data read attempt. Can be dcraw, pamcut, netPbm... :')
            logger.error(e)
            raise e
    
    else:
        logger.critical('shit went wrong')
        raise NotImplementedError('shit went wrong')

    try:
        if inverseGrayscale:
            maxBitPix = np.ndarray.max(im_ppm)
            im_ppm = maxBitPix * np.ones_like(im_ppm) - im_ppm

        # Create the FITS object
        hdu = pyfits.PrimaryHDU(im_ppm)
        
        
        
        if camera in SATURATION_VALUE_CATALOG:
            hdu.header.set('DATAMAX', SATURATION_VALUE_CATALOG[camera],
                           'pixel values above this level are considered saturated')

        # set basic header values
        hdu.header.set('OBSTIME', exposure_corrected_time.iso, 'ISO timestamp of start of exposure')
        #hdu.header.set('OBSTIME', "2021-08-06 21:12:10", 'ISO timestamp of start of exposure')
        # comment next line to bypass shutter fraction problem
        hdu.header.set('EXPTIME', float(shutter), 'Exposure time in s')
        hdu.header.set('APERTUR', float(aperture), 'Lens aperture, as in f/APERTUR')
        hdu.header.set('ISO', int(iso), 'ISO speed')
        hdu.header.set('FOCAL', float(focal), 'Focal length in mm')
        hdu.header.set('FILT_PAT', filter_pattern, 'Shape of the Color array (usually Bayer)')
        hdu.header.set('ORIGIN', original_file)
        hdu.header.set('CAMERA', camera, 'Camera model')
        hdu.header.add_comment('FITS File Created with ' + __pipeline_task__)
        if needsCropping:
            hdu.header.set('CROPPED', "TRUE", 'If picture has been cropped from original')
            hdu.header.set('CROP', cropprofile, 'Crop preset (for further use)')
            hdu.header.set('CROPX0', x0, 'LEFT boundary of the crop')
            hdu.header.set('CROPX1', x0 + x_size, 'RIGHT boundary of the crop')
            hdu.header.set('CROPY0', y0, 'BOTTOM boundary of the crop')
            hdu.header.set('CROPY1', y0 + y_size, 'TOP boundary of the crop')
            hdu.header.set('ONAXIS1', NAXIS1, 'Original NAXIS1 before cropping')
            hdu.header.set('ONAXIS2', NAXIS2, 'Original NAXIS2 before cropping')
        else:
            hdu.header.set('CROPPED', "FALSE", 'If picture has been cropped from original')
    except:
        logger.error('Something went wrong while creating the FITS object')
        raise

    logger.info("RAW extraction into FITS successful")

    return hdu


def add_meta(rawHDU, station_config_input='', supplargs={}):
    """
    Update FITS header with exposure information
    parameters:
        - rawHDU: hdu object to update
        - station_config_input: dfnstation.cfg config file path, or pre-loaded equivalent of that file (into a ConfigParser object)
        - supplargs: dictionary of other FITS card to add
    returns:
        - nothing (just updates the hdu object)
    """
    
    logger = logging.getLogger()
    
    rawHeader = rawHDU.header
    
    # add missing header FITS info (cf metadata_guidebook document)
    obsTimeNonISO = rawHeader['OBSTIME']

    focal = rawHeader['FOCAL']
    # test what type of lens it is, based on focal, not really good code, but essential to be able to process any image automatically
    if focal < 13:
        lenstype = "Fisheye"
    else:
        lenstype = "Rectilinear"
        

    [rawHeader.set(k, v) for k, v in list(supplargs.items())]

    # replace the blank space with 'T' to comply with ISO 8601 date
    rawHeader.set('DATE-OBS', obsTimeNonISO.replace(" ", "T", 1), 'date of the observation')
    rawHeader.set('INSTRUME', rawHeader['CAMERA'])
    rawHeader.set('DETNAM', rawHeader['CAMERA'], 'Name of the detector used to make the observation')
    rawHeader.set('LENSTYPE', lenstype, 'Lens design')
    rawHeader.set('ISOSPEED', rawHeader['ISO'], 'ISO settings the camera is running on')
    rawHeader.set('CREATOR', __pipeline_task__ + " " + __version__, 'the name of the software task that created the file')
    rawHeader.set('AUTHOR', __author__, 'identifies who compiled the information in the data associated with the header')
    rawHeader.set('ORIGIN', "Desert Fireball Network", "Organization responsible for the data")
    rawHeader.set('OBSERVER', "DFN automated observatory http://adsabs.harvard.edu/abs/2017ExA....43..237H", "Observer who acquired the data")

    if isinstance(station_config_input, ConfigParser):
        Config = station_config_input
    elif station_config_input != '' and station_config_input != None and os.path.isfile(station_config_input):  # None is the Null object in Python
        # now parse the station data file (cf dfnstation.cfg document)
        Config = ConfigParser()
        Config.read(station_config_input)
        
        try:
            gps_lock = Config.get('station', 'gps_lock')
        except configparser.NoOptionError:
            gps_lock = 'N'
        except:
            gps_lock = 'N'
        #if gps_lock == 'Y':
        # Accomodates flavors of lock flag:
        # - no flag: very early systems 2013 -> 2014
        # - Y/N: DFNSMALL 2014 -> 2016
        # - 0-8: switch to GGA sentence end 2016
        # - N/Yn: mix introduced early 2017 to keep Y/N, but record GGA status code
        # see leostick.py for details on GGA sequence code
        if gps_lock.startswith('Y') or (gps_lock.isnumeric() and int(gps_lock) >= 1):
            rawHeader.set('GPSLOCK', 'Y', 'Camera has reliable GPS signal')
            logger.info("GPS lock confirmed")
            if len(gps_lock) > 1 and gps_lock.startswith('Y'):
                gga_code = gps_lock[1:]
                rawHeader.set('GPSGGA', gga_code, 'GPS GGA sentence status code')
                logger.debug("GPS GGA sentence read")
        else:
            rawHeader.set('GPSLOCK', 'N', 'Camera has reliable GPS signal')
            logger.warning("NO GPS lock")
        rawHeader.set('SITELAT', float(Config.get('station', 'lat')))
        rawHeader.set('SITELONG', float(Config.get('station', 'lon')))
        rawHeader.set('SITEALT', float(Config.get('station', 'altitude')))
        rawHeader.set('LENS', Config.get('camera', 'still_lens'))
        
    # depending on the dfnstation.cfg file version, station name is either "name" or "hostname"
        try:
            thestationname = Config.get('station', 'name')
        except configparser.NoOptionError:
            thestationname = Config.get('station', 'hostname')
        rawHeader.set('SITEDFN', thestationname)
        rawHeader.set('TELESCOP', thestationname, 'Name of telescope')
        rawHeader.set('SITELOC', Config.get('station', 'location'))
        if Config.has_option('station', 'tracking') and (Config.get('station', 'tracking') == "YES"):
            rawHeader.set('TRACKIN', 'YES', 'Mount in tracking mode')
        else:
            rawHeader.set('TRACKIN', 'NO', 'Mount in tracking mode')
    else:
        rawHeader.set('GPSLOCK', 'N')

    rawHeader.set('BITPIX', 16, 'bits per data value')
    rawHeader.set('DATE', datetime.datetime.utcnow().isoformat('T'), 'file creation UTC timestamp')


def de_Bayer(rawHDU, color_code='green'):
    """
    De_Bayer function
    parameters:
        - rawHDU: raw FITS hdu
        - color_code:
                        green = green interpolated
                        green1, green2, red, blue = collapse rows/columns on other pixels
                        rgb2x2 = 2X2 binning mixing all colors
                        green2x2 = 2X2 binning mixing only the 2 green pixels in the pattern
    returns:
        - hdu of de-Bayered image
    """
    
    logger = logging.getLogger()
    
    logger.debug('Attempting de_Bayer using the {0} method'.format(color_code))
    
    # read image size from header
    rawHeader = rawHDU.header
    width = rawHeader['NAXIS1']
    height = rawHeader['NAXIS2']
    
    # read image data
    rawIMG = rawHDU.data
    
    # create output data frames
    de_Bayered_data = np.zeros((height, width), dtype="uint16")
    de_Bayered_header = deepcopy(rawHeader)
    
    # Prepare rows/columns collapsing masks
    # Odd/even rows
    allCols = np.arange(1, width + 1, 1)  # HD: overrides python default table start at 0
    allRows = np.arange(1, height + 1, 1)
    oddCols = allCols % 2 == 1
    # print "length: "+ str(len(oddCols))
    # print oddCols[:10:1] # HD: [ True False  True False  True False ... ] and True=1 / False=0
    evenCols = allCols % 2 == 0
    oddRows = allRows % 2 == 1
    evenRows = allRows % 2 == 0
    
    # swap accordingly
    # The assumed pattern looks like that:
    #   R | G2
    #   ------
    #   G1| B
    # method to know bayer pattern:
    # set camera to low iso, completely defocus, take a picture of a red surface in raw.
    # dcraw -D -4 -c image_name.NEF  | pnmtofits > red.fits
    # aladin red.fits
    # look at the top left corner, and look where the bright pixel is located, this is your first left pixel. or look at "filter pattern" using dcraw -v -i <image>
    #   if camera_type == 'Nikon D810' or camera_type == 'Nikon D800' or camera_type == 'Nikon D800E':
    filt_pat = rawHeader['FILT_PAT'].replace(" ", "")
    if filt_pat == "RGGBRGGBRGGBRGGB" or filt_pat == 'RG/GB':
        logger.info("Camera type is supported for RGB splitting")
    else:
        logger.warning("Color pattern may be wrong, DSLR type is not supported. Assuming RGGBRGGBRGGBRGGB")
        
    
    # Step 1: copy data in appropriate pixel slots
    # Step 2: Collapse each array by deleting the correct rows/columns, dependending on Bayer pattern, proper to each sensor
    # Step 3: add FITS card to log de-Bayering technique used

    # no need to de-bayer if pattern required is raw
    if color_code == 'raw':
        de_Bayered_data = rawIMG
        de_Bayered_header.set('COLORTYP', 'RAW', 'De-Bayer processing filter: NOT de-Bayered')
    elif 'red' == color_code:
        de_Bayered_data[::2, ::2] = rawIMG[::2, ::2]       # RED
        de_Bayered_data = de_Bayered_data[oddRows]
        de_Bayered_data = de_Bayered_data[:, oddCols]
        de_Bayered_header.set('COLORTYP', 'RED', 'De-Bayer processing filter: Only keep R from RGGB block')

    elif 'green1' == color_code:
        de_Bayered_data[::2, 1::2] = rawIMG[::2, 1::2]     # GREEN 1
        de_Bayered_data = de_Bayered_data[oddRows]
        de_Bayered_data = de_Bayered_data[:, evenCols]
        de_Bayered_header.set('COLORTYP', 'GREEN 1', 'De-Bayer processing filter: Only keep G1 from RGGB block')

    elif 'green2' == color_code:
        de_Bayered_data[1::2, ::2] = rawIMG[1::2, ::2]     # GREEN 2
        de_Bayered_data = de_Bayered_data[evenRows]
        de_Bayered_data = de_Bayered_data[:, oddCols]
        de_Bayered_header.set('COLORTYP', 'GREEN 2', 'De-Bayer processing filter: Only keep G2 from RGGB block')

    elif 'blue' == color_code:
        de_Bayered_data[1::2, 1::2] = rawIMG[1::2, 1::2]    # BLUE
        de_Bayered_data = de_Bayered_data[evenRows]
        de_Bayered_data = de_Bayered_data[:, evenCols]
        de_Bayered_header.set('COLORTYP', 'BLUE', 'De-Bayer processing filter: Only keep B from RGGB block')

    elif 'green' == color_code:
        de_Bayered_data[1::2, ::2] = rawIMG[1::2, ::2]     # GREEN 2
        de_Bayered_data[::2, 1::2] = rawIMG[::2, 1::2]     # GREEN 1
        # interpolation with neighbouring pixels
        de_Bayered_data[2::2, 2::2] = (rawIMG[2::2, 1:width - 2:2] +
                                rawIMG[2::2, 3::2] +
                                rawIMG[1:height - 2:2, 2::2] +
                                rawIMG[3::2, 2::2]) / 4    # RED -> GREEN
        de_Bayered_data[1:height - 2:2, 1:width - 2:2] = (rawIMG[:height - 2:2, 1:width - 2:2] +
                                                   rawIMG[2:height:2, 1:width - 2:2] +
                                                   rawIMG[1:height - 2:2, :width - 2:2] +
                                                   rawIMG[1:height - 2:2, 2:width:2]) / 4     # BLUE -> GREEN
        de_Bayered_header.set('COLORTYP', 'GREEN_INTERPOL', 'De-Bayer processing filter: Full resolution image on green channel, missing pixel interpolated with nearest green neighbors')

    elif 'rgb2x2' == color_code:
        de_Bayered_data[::2, ::2] = (rawIMG[::2, ::2] +  # RED
                                   rawIMG[::2, 1::2] +     # GREEN 1
                                   rawIMG[1::2, ::2] +     # GREEN 2
                                   rawIMG[1::2, 1::2])    # BLUE
        de_Bayered_data = de_Bayered_data[oddRows]
        de_Bayered_data = de_Bayered_data[:, oddCols] 
        de_Bayered_header.set('COLORTYP', 'RGB_2X2', 'De-Bayer processing filter: Bin RGGB block') 
        de_Bayered_header.set('XBINNING', 2)
        de_Bayered_header.set('YBINNING', 2)

    elif 'green2x2' == color_code:
        de_Bayered_data[::2, ::2] = (rawIMG[::2, 1::2] +     # GREEN 1
                                 rawIMG[1::2, ::2])     # GREEN 2
        de_Bayered_data = de_Bayered_data[oddRows]
        de_Bayered_data = de_Bayered_data[:, oddCols]
        de_Bayered_header.set('COLORTYP', 'GREEN_2X2', 'De-Bayer algo: Bin RGGB block using 2 Gs only')
        de_Bayered_header.set('XBINNING', 2)
        de_Bayered_header.set('YBINNING', 2)

    else:
        raise KeyError('De-Bayering method {0} not found'.format(color_code))
    
    logger.debug('De-Bayer successful')

    return pyfits.PrimaryHDU(data=np.flipud(de_Bayered_data),
                             header=de_Bayered_header)


def output_filename(raw_file, color_code, out_dir=None, croptextinfilename=""):
    """
    Generate output file names for FITS image
    parameters:
        - raw_file: raw file path
        - color_code: 
        - out_dir (optional): specify different output directory
        - croptextinfilename (optional): suffix to add to the filename
    returns:
        - file name
    """
    # get filename without extension
    
    if out_dir:
        file_name = os.path.join(out_dir, os.path.splitext(os.path.basename(raw_file))[0])
    else:
        file_name = os.path.splitext(raw_file)[0]
        
      
    if 'raw' == color_code:
        color_suffix = "raw"
    elif 'red' == color_code:
        color_suffix = "R"
    elif 'green1' == color_code:
        color_suffix = "G1"
    elif 'green2' == color_code:
        color_suffix = "G2"
    elif 'blue' == color_code:
        color_suffix = "B"
    elif 'green' == color_code:
        color_suffix = "G"
    elif 'rgb2x2' == color_code:
        color_suffix = "RGB2X2"
    elif 'green2x2' == color_code:
        color_suffix = "G2X2"
    else:
        raise KeyError('Color code {0} not found'.format(color_code))
    
    file_name += croptextinfilename + "-" + color_suffix + "." + FITS_EXTENSION
    
    return file_name


def tiler(img):
    
    
    if 'supertiler_' in crop:
            crop_matrix = crop.split('_')[1]
            numtiles_longaxis = int(crop_matrix.split('x')[0])
            numtiles_shortaxis = int(crop_matrix.split('x')[1])
            
            crop_size = int(crop.split('_')[1])
            # top left pixel must have even coordinates, otherwise bayer pattern will be shifted
            x0 = int((NAXIS1 - crop_size) / 2)
            y0 = int((NAXIS2 - crop_size) / 2)
            y_size = crop_size
            x_size = crop_size
            cropprofile = crop
    
    
    im_ppm = raw_data[y0:y0+y_size, x0:x0+x_size]



#def crop():
    #centered_
    #supertiler_6x4
    
   
        #needsCropping = True
        ## regex for finding 4 numbers separated by commas
        #match = re.search("[0-9]+(,[0-9]+){3}", crop)
        ## Crop centered presets
        #if 'centered_' in crop and crop.split('_')[1].isnumeric():
            #crop_size = int(crop.split('_')[1])
            ## top left pixel must have even coordinates, otherwise bayer pattern will be shifted
            #x0 = int((NAXIS1 - crop_size) / 2)
            #y0 = int((NAXIS2 - crop_size) / 2)
            #y_size = crop_size
            #x_size = crop_size
            #cropprofile = crop
        #if 'supertiler_' in crop:
            #crop_matrix = crop.split('_')[1]
            #numtiles_longaxis = int(crop_matrix.split('x')[0])
            #numtiles_shortaxis = int(crop_matrix.split('x')[1])
            
            #crop_size = int(crop.split('_')[1])
            ## top left pixel must have even coordinates, otherwise bayer pattern will be shifted
            #x0 = int((NAXIS1 - crop_size) / 2)
            #y0 = int((NAXIS2 - crop_size) / 2)
            #y_size = crop_size
            #x_size = crop_size
            #cropprofile = crop
        #elif match:
            #cropstr = match.group().split(',')
            #x0 = int(cropstr[0])
            #y0 = int(cropstr[1])
            #x_size = int(cropstr[2])
            #y_size = int(cropstr[3])
            #cropprofile = "Custom " + crop
        #else:
            #needsCropping = False
            #cropprofile = "None"
            #logger.error('Crop input is not valid. Crop aborted')
            #raise KeyError('Invalid crop parameters')

        ## top left pixel must have even coordinates, otherwise bayer pattern will be shifted
        #if x0 % 2 == 1:
            #x0 += 1
        #if y0 % 2 == 1:
            #y0 += 1

        ## check the requested crop falls into the picture bounds
        #if (x0 >= NAXIS1) or (y0 >= NAXIS2) or ((x0 + x_size) >= NAXIS1) or ((y0 + y_size) >= NAXIS2) or (x0 < 0) or (y0 < 0):
            #needsCropping = False
            #cropprofile = "None"
            #OOB_msg = 'Crop input does NOT fall into picture bounds. Crop aborted'
            #logger.error(OOB_msg)
            #raise IndexError(OOB_msg)

        ## use netpbm's pamcut to crop
        #crop_command = ['pamcut', '-left', str(x0), '-top', str(y0), '-width', str(x_size), '-height', str(y_size)]
    
    #im_ppm = raw_data[y0:y0+y_size, x0:x0+x_size]
    
    
    
    #if needsCropping:
        #hdu.header.set('CROPPED', "TRUE", 'If picture has been cropped from original')
        #hdu.header.set('CROP', cropprofile, 'Crop preset (for further use)')
        #hdu.header.set('CROPX0', x0, 'LEFT boundary of the crop')
        #hdu.header.set('CROPX1', x0 + x_size, 'RIGHT boundary of the crop')
        #hdu.header.set('CROPY0', y0, 'BOTTOM boundary of the crop')
        #hdu.header.set('CROPY1', y0 + y_size, 'TOP boundary of the crop')
        #hdu.header.set('ONAXIS1', NAXIS1, 'Original NAXIS1 before cropping')
        #hdu.header.set('ONAXIS2', NAXIS2, 'Original NAXIS2 before cropping')
    #else:
        #hdu.header.set('CROPPED', "FALSE", 'If picture has been cropped from original')


def main(rawfile,
         outDir,
         doAstrometry=False,
         cropparam='',
         cfgstationfile='',
         omode="green",
         masterdark="",
         masterflat="",
         pedantic=False,
         inverseGrayscale=False,
         tiff_conversion=False,
         jpeg_conversion=False):
    """
    Generate output file names for FITS image
    parameters:
        - rawfile: raw file path
        - outDir: specify different output directory
        - doAstrometry: calculate astromatic solution (NOT IMPLEMENTED)
        - cropparam: crop parameters
        - cfgstationfile: path to dfnstation.cfg file
        - omode: de-bayer mode
        - masterdark: path to dark frame
        - masterflat: path to flat field
        - pedantic: fail if some key metadata is absent
        - tiff_conversion: convert output to tiff (requires stiff)
        - jpeg_conversion: convert output to jpeg
    returns:
        - list of FITS files created
    """
    
    logger = logging.getLogger()
    
    dirName = os.path.dirname(rawfile)
    
    if doAstrometry:
        raise NotImplementedError('Astrometry not implemented')
    
    
    dfnstation_file_valid = False
    if cfgstationfile != '':
        if os.path.isfile(cfgstationfile):
            dfnstation_file_valid = True
            logger.info("Using " + dfn_utils.stdcfgfilename + " file specified")
        else:
            raise FileNotFoundError('Cannot find given config file: ' + cfgstationfile)
    else:
        # try to find one
        cfgstationfile = os.path.join(dirName, dfn_utils.getDfnstationConfigFile(dirName))
        if os.path.isfile(cfgstationfile):
            dfnstation_file_valid = True
            logger.info("Using " + dfn_utils.stdcfgfilename + " file found")

    if pedantic and not dfnstation_file_valid:
        raise ConfigFileNotFoundError('Cannot find mandatory station config file')
    elif not dfnstation_file_valid:
        logger.warning(dfn_utils.stdcfgfilename + " file not provided nor found, expect LOCATION data problems!")


        
        
    needscropping = (cropparam != "")
    if needscropping:
        croptextinfilename = "-crop"
    else:
        croptextinfilename = ""
        
    
    
    if omode == "greenraw":
        ocolorfiles = ['green', 'raw']
    if omode == "colors":
        ocolorfiles = ['green', 'green1', 'green2', 'blue', 'red']
    if omode == "raw":
        ocolorfiles = ['raw']
    if omode == "green":
        ocolorfiles = ['green']
    if omode == "red":
        ocolorfiles = ['red']
    if omode == "blue":
        ocolorfiles = ['blue']
    if omode == "full":
        ocolorfiles = ['green', 'green1', 'green2', 'blue', 'red', 'raw']
    if omode == "rgb2x2":
        ocolorfiles = ['rgb2x2']
    if omode == "green2x2":
        ocolorfiles = ['green2x2']

    
    if rawfile.lower().endswith('.thumb.jpg'):
        logger.warning('We are dealing with a source JPG thumb file, not the same resolution!')
    
    logger.info('Starting RAW extraction...')
    rawhdu = raw2fits(rawfile, cropparam, dark=masterdark, flat=masterflat, inverseGrayscale=inverseGrayscale)
    

    # add additional header data (imagetype...)
    supplargslist = []
    supplargslist.append(("IMAGETYP", "LIGHT"))
    supplargslist.append(("FILENAME", os.path.basename(rawfile)))
    supplargslist.append(("INPUTFMT", os.path.basename(rawfile).split('.')[-1]))
    # IRAF standard: "LIGHT", "BIAS", "FLAT" and "DARK"

    if cfgstationfile != '':
        supplargslist.append(("LOCINFO", "YES"))
    else:
        supplargslist.append(("LOCINFO", "NO"))

    # comprehension list version
    supplargs = {key: value for (key, value) in supplargslist}
    
    
    # add all missing metadata
    add_meta(rawhdu, cfgstationfile, supplargs)

    if cfgstationfile != '':
        logger.debug("Location info added to header")
    else:
        logger.warning("Image does NOT contain camera Earth location coordinates, not ready for calibrating!")
    logger.debug("General info added to header")
    
    
    
    if inverseGrayscale:
        maxBitPix = np.ndarray.max(rawhdu.data)
        rawhdu.data = maxBitPix * np.ones(np.shape(rawhdu.data), dtype="uint16") - rawhdu.data

    fits_files_created = []
    for color_code in ocolorfiles:
        be_bayer_fits_object = de_Bayer(rawhdu, color_code=color_code)
        
        file_name = output_filename(rawfile, color_code, croptextinfilename=croptextinfilename, out_dir=outDir)
        
        #pyfits.writeto(file_name, be_bayer_fits_object)
        try:
            be_bayer_fits_object.writeto(file_name, overwrite=True, output_verify="silentfix")
        except TypeError:
            # for astropy < 1.3
            be_bayer_fits_object.writeto(file_name, clobber=True, output_verify="silentfix")
            
        fits_files_created += [file_name]
        
        logger.info('FITS files created: {0}'.format(file_name))

    

    # if Tiff output is required
    if tiff_conversion:
        logger.debug("Launching STIFF to convert all output FITS files to TIFF")
        for ofile in fits_files_created:
            convertToTIFF(ofile)
        if jpeg_conversion:
            logger.debug("Converting outputs to jpegs")
            for ofile in fits_files_created:
                otifffile = os.path.splitext(ofile)[0] + "." + dfn_utils.tifExtension
                ojpgfile = os.path.splitext(ofile)[0] + "." + dfn_utils.jpgExtension
                os.system("convert " + otifffile + " " + ojpgfile)

    return fits_files_created


if __name__ == "__main__":
    import argparse
    import timeit
    scp_start_time = timeit.default_timer()
    
    # stdout logger
    logger = logging.getLogger()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(module)s, %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    # parse arguments
    parser = argparse.ArgumentParser(description='Create FITS images from raw file, add necessary fits header data. Compatible cameras are the same as your version of dcraw.')
    inputgroup = parser.add_mutually_exclusive_group(required=True)
    inputgroup.add_argument("-i", "--inputfile", type=str,
                            help="input raw images file name")
    inputgroup.add_argument("-d", "--inputdirectory", type=str,
                            help="input directory for images with extension " + DEFAULT_RAW_EXTENSION)
    parser.add_argument("-c", "--stationcfgfile", type=str, default="",
                        help="station config file name (if not provided, will try to find a file called dfnstation.cfg in the same directory as input)")
    parser.add_argument("-O", "--outputdir", type=str,
                        help="output directory (same as input if not specified)", default=None)
    parser.add_argument("-K", "--masterdark", type=str, default="",
                        help="PGM preprocessing image to subtract, see preProcessImage.py or https://www.cybercom.net/~dcoffin/dcraw/dcraw.1.html")
    parser.add_argument("-F", "--masterflat", type=str, default="",
                        help="")
    parser.add_argument("-z", "--crop", type=str, default="",
                        help="Use this option if you want to crop the image. Presets: centered_XXXX, supertiler_12. Or custom <x0,y0,size_x,size_y>")
    parser.add_argument("-o", "--outputmode", type=str,
                        help="Output mode (greenraw (only raw and green), green (only green interpolated, default), full, colors, raw, interpol, red, blue)",
                        choices=['green',
                                 'full',
                                 'colors',
                                 'greenraw',
                                 'raw',
                                 'red',
                                 'blue',
                                 'rgb2x2',
                                 'green2x2'],
                        default="green")
    parser.add_argument("-t", "--tiff", action="store_true", default=False,
                        help="use this option if you want to get TIFF on top of the FITS files. Requires STIFF (http://www.astromatic.net/software/stiff)")
    parser.add_argument("-j", "--jpeg", action="store_true", default=False,
                        help="use this option if you want to get lossy 255 bits JPEGs")
    parser.add_argument("-T", "--tempfolder", action="store_true", default=False,
                        help="Uses " + TEMP_DIR + " for processing (faster if data is on remote drive, or slow mechanical drive)")
    parser.add_argument("-r", "--rawextension", type=str,
                        help="Raw extension", default=DEFAULT_RAW_EXTENSION)
    parser.add_argument("-a", "--astrometry", action="store_true", default=False,
                        help="Compute astrometry (will try to figure out best parameters to do it). NOT IMPLEMENTED")
    parser.add_argument("-b", "--whitebackground", action="store_true", default=False,
                        help="Inverse grayscale")
    parser.add_argument("-p", "--pedantic", action="store_true", default=False,
                        help="Will complain about lack of dfnstation.cfg file")
    parser.add_argument("-l", "--decodinglibrary", type=str, default=None,
                        help="force decoding library (dcraw, rawpy)")

    args = parser.parse_args()
    
    #print(vars(args))

    rawExtension = args.rawextension

    doAstrometry = args.astrometry
    
    tiff_conversion = args.tiff

    inverseGrayscale = args.whitebackground
    
    cropparam = args.crop

    masterdark = args.masterdark
    
    masterflat = args.masterflat
    
    pedantic = args.pedantic

    omode = args.outputmode
    
    if args.decodinglibrary:
        if args.decodinglibrary.lower() == 'dcraw':
            import netPbm
            RAW_HANDLING_LIB = 'DCRAW'
        elif args.decodinglibrary.lower() == 'jpg':
            RAW_HANDLING_LIB = 'JPG'

    # jpeg output needs the tiff
    jpegOutput = args.jpeg
    if jpegOutput and not tiff_conversion:
        tiff_conversion = True

    # Get directory listings

    dirList = []

    if args.inputfile is not None:
        dirList.append(args.inputfile)
        dirName = os.path.dirname(args.inputfile)
    elif os.path.isdir(args.inputdirectory):
        directory = args.inputdirectory
        # list all raw files in this directory
        dirList = dfn_utils.resolve_glob(rawExtension, directory=directory, prefix="")
        if len(dirList) < 1:
            logger.error('No image with extension ' + rawExtension + ' found in folder ' + directory)
            exit(1)
        print(dirList)
        dirName = directory
    else:
        logger.error('Input file (--inputfile) or directory (--inputdirectory) invalid')
        exit(1)

    # test write rights in working directory
    # commented out by MCT, as problems with the NAS over network mount
    # if not os.access(dirName, os.W_OK):
    #    print dfn_utils.fail, __pipeline_task__, " does NOT have write permissions in ",dirName
    #    exit(1)
    
    
    cfgstationfile = args.stationcfgfile


    # Output dir
    if args.outputdir is not None:
        outDir = args.outputdir
    else:
        outDir = dirName

    numberoffiles = len(dirList)
    currentfile = 0
    
    for i in dirList:
        currentfile += 1
        logger.info('Processing image {0:d}/{1:d} : {2})'.format(currentfile, numberoffiles, os.path.basename(i)))


        main(rawfile=i,
             outDir=outDir,
             omode=omode,
             doAstrometry=doAstrometry,
             cropparam=cropparam,
             cfgstationfile=cfgstationfile,
             masterdark=masterdark,
             masterflat=masterflat,
             pedantic=pedantic,
             inverseGrayscale=inverseGrayscale,
             tiff_conversion=tiff_conversion,
             jpeg_conversion=jpegOutput)
        
        
    logger.debug("%s image(s) converted in %.2fs", currentfile, timeit.default_timer() - scp_start_time)
    logger.debug("Process exited normally")
    
    
