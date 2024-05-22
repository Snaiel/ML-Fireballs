#!/usr/bin/env python
#
#
# Version history:
# 



__author__ = "Hadrien A.R Devillepoix"
__copyright__ = "Copyright 2018, Desert Fireball Network"
__license__ = "MIT"
__version__ = "1.0"


# files extensions
FITS_EXTENSION = 'fits'
TIF_EXTENSION = 'tiff'
DEFAULT_RAW_EXTENSION = 'NEF'


# delay between trigger and actual shutter open on digital cameras (in seconds)
DSLR_TRIGGER_DELAY_CATALOG = {'Nikon D800E': 0.060,
                              'Nikon D800': 0.060,
                              'Nikon D810': 0.056,
                              'Nikon D850': 0.060,
                              'Nikon D750': 0.060,
                              'Sony ILCE-7SM2': 0.060}



RAW_EXTENSIONS = {'NEF' : 'nikon',
                  'CR2' : 'canon',
                  'ARW' : 'sony'}


# Can add lenses here, use lower case
# equisolid
# equidistance
# orthogonal
# stereographic
FISHEYE_LENS_PROJECTION_CATALOG = {'samyang_8mm_f3.5':'stereographic'}


# saturation pixel values
SATURATION_VALUE_CATALOG = {'Nikon D800E': 16383,
                       'Nikon D800': 16383,
                       'Nikon D810': 16350}

# in millimeters and pixels
SENSOR_SIZE_CATALOG = {'Nikon D800E': (36.0, 24.0, 7424, 4924),
                       'Nikon D800': (36.0, 24.0, 7424, 4924),
                       'Nikon D810': (36.0, 24.0, 7380, 4928),
                       'Nikon D850': (36.0, 24.0, 8288, 5520),
                       'Nikon D750': (36.0, 24.0, 6032, 4032),
                       'Sony ILCE-7SM2': (36.0, 24.0, 4256, 2848),
                       'ZWO ASI1600MM Pro': (18.0, 13.5, 4656, 3520)}


IMAGE_FULL_VS_THUMB_CATALOG = {'Nikon D800E': (7424, 4924, 7360, 4912),
                       'Nikon D800': (7424, 4924, 7360, 4912),
                       'Nikon D810': (7380, 4928, 7360, 4912),
                       'Nikon D850': (8288, 5520, 0, 0),
                       'Nikon D750': (6032, 4032, 0, 0),
                       'Sony ILCE-7SM2': (4256, 2848, 0, 0)}
