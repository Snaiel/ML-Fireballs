#!/usr/bin/env python
#
# Version history:
# 1.0	Initial commit - R Howie
# 1.1	Added fits header reading for meta - H Devillepoix
# 1.3	Added velocity plots key - H Devillepoix
# 1.4	Time decoding using external module timing_corrections - H Devillepoix
# 1.5	Time not decodable flag - H Devillepoix
# 1.6	Bug fixes - H Devillepoix
# 2.0	Python 3, finally! - H Devillepoix


# general modules
import argparse
import datetime
import os
import sys
import socket
import getpass

# import science modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RadioButtons, Slider
from math import sqrt
from astropy.table import Table, Column
from astropy import units as u
import astropy.io.fits as pyfits
#from scipy.interpolate import BSpline

# import custom modules
import de_bruijn
import dfn_utils
import timing_corrections
import parseObserver
import technical_constants_DFN

__author__ = "Robert Howie, Hadrien Devillepoix"
__copyright__ = "Copyright 2015-2022, Desert Fireball Network"
__license__ = "MIT"
__version__ = "3.0"
__scriptName__ = "point_picker.py"

GFE_standard_version = 1.1


# globals
db_seq = de_bruijn.de_bruijn_sequence(2, 9).de_bruijn_sequence
IMAGE_VIEWER = 'open'

#+/- y&x picking error in pix
picking_error_pix = 0.5
default_timing_error_minus_s = 0.65e-3
default_timing_error_plus_s = 0.05e-3

# fits extension
fitsExtension = "fits"


if not sys.version_info[0] < 3.7:
    print("Need python {}".format(">= 3.7"))
    exit(1)


# point class
class point:

    def __init__(self, x, y, time_err_minus, time_err_plus, sequence_index, start, circle_instance, firmware='april2014'):
        self.x = x
        self.x_err_minus = picking_error_pix
        self.x_err_plus = picking_error_pix
        self.y = y
        self.y_err_minus = picking_error_pix
        self.y_err_plus = picking_error_pix
        self.time_err_minus = time_err_minus
        self.time_err_plus = time_err_plus
        self.sequence_index = sequence_index
        self.start = start
        self.start = start

        # circle instance is a matplotlib artist that draws the dot on the graph
        self.circle_instance = circle_instance

        # periodic pulse is for future PFSK data
        self.periodic_pulse = '-'
        #'M' for manually picked
        self.pick_flag = 'M'
        #'PW' pulse width, 'PF' pulse frequency
        if firmware == 'april2014':
            self.encoding_type = 'PW'
        elif firmware == 'june2017':
            self.encoding_type = 'PF'
        elif firmware == 'boogardie_PW_1_0_continuous' or firmware == 'oukaimeden_PW_ends_only':
            self.encoding_type = 'PW_1_0_continuous'
        else:
            raise timing_corrections.UnknownEncodingError('')

        #self.datetime = datetime

    # useful for printing point info if required for debugging
    def __str__(self):
        time = self.sequence_index * 0.1 + 0.02 * (not self.start) + db_seq[self.sequence_index] * 0.04
        if self.start == 'S':
            start_end = 'start'
        else:
            start_end = 'end'
        return '{0}, {1}, {2:.2f} #{3} {4}'.format(self.x, self.y, time, self.sequence_index, start_end)

# point picker
# http://matplotlib.org/users/event_handling.html


class picker:

    def __init__(self,
                 image_path,
                 previous_points_path=None,
                 pointPickingComment="rough",
                 fireballName="fireball",
                 scaling=1.0,
                 startSeqIndex=0,
                 shutter_inversion=False,
                 pick_start_and_end_exposure=False):
        def cmap_func(label):
            self.imgplot.set_cmap(label)
            self.fig.canvas.draw()
            
        def scales_func(val):
            self.imgplot.set_clim(vmin=self.vmin.val, vmax=self.vmax.val)
            self.fig.canvas.draw()
                
        self.pick_start_and_end_exposure = pick_start_and_end_exposure

        # create figure
        self.fig = plt.figure()
        # adjust bottom margin to leave room for sequence info and preview
        self.fig.subplots_adjust(bottom=0.15)
        # create some axes
        self.ax = self.fig.add_subplot(111)
        self.user_comment = pointPickingComment

        # image directory and filename handling
        if os.path.realpath(image_path) != image_path:
            print('One of the path given is not an absolute one, or contains a symlink. Exiting...')
            exit(2)
        self.image_path = image_path
        self.image_directory = os.path.dirname(self.image_path)
        if self.image_directory == '':
            self.image_directory = '.'
        self.image_basename = os.path.basename(self.image_path)
        self.image_filename = os.path.splitext(self.image_basename)[0]

        self.shutter_inversion = shutter_inversion

        # Guess fireball codename if passed argument is set to default
        if fireballName in ['fireball', '']:
            try:
                found_codename = dfn_utils.extract_event_codename_from_path(image_path)
                self.eventcodename = found_codename
            except ValueError:
                self.eventcodename = fireballName
            
        # create path base for output
        self.output_filename = os.path.join(self.image_directory, self.image_filename)
        
        # load up the image
        _, file_extension = os.path.splitext(image_path)
        if file_extension == '.fits':
            input_image_type = 'FITS'
            hdulist = pyfits.open(image_path, memmap=False)
            img = np.flipud(hdulist[0].data)
            prihdr = hdulist[0].header
        elif file_extension == '.' + technical_constants_DFN.DEFAULT_RAW_EXTENSION:
            input_image_type = 'RAW'
            # RAW image
            # first determine if cfg file is there
            cfg = os.path.join(os.path.dirname(image_path), 'dfnstation.cfg')
            if not os.path.isfile(cfg):
                raise FileNotFoundError("Missing dfnstation.cfg config file alongside the input image")
            import fitsConversion
            # read the raw file
            rawHDU = fitsConversion.raw2fits(image_path)
            
            # add additional header data (imagetype...)
            supplargslist = []
            supplargslist.append(("IMAGETYP", "LIGHT"))
            supplargslist.append(("FILENAME", os.path.basename(image_path)))
            supplargslist.append(("INPUTFMT", os.path.basename(image_path).split('.')[-1]))
            supplargslist.append(("LOCINFO", "YES"))
            supplargs = {key: value for (key, value) in supplargslist}
            fitsConversion.add_meta(rawHDU, station_config_input=cfg, supplargs=supplargs)
            
            # de-bayer
            hdu = fitsConversion.de_Bayer(rawHDU, color_code='green')
            
            img = np.flipud(hdu.data)
            prihdr = hdu.header
            
            # correct basename to add the "-G" suffix
            self.output_filename += "-G"
        else:
            input_image_type = 'other'
            img = mpimg.imread(image_path)
            # create image metadata from FITS
            # read fits header. usually image.tiff is stored with a file image.fits, which contains a header with all necessary metadata
            self.fitsimage_filename = self.output_filename + "." + fitsExtension
            if os.path.isfile(self.fitsimage_filename):
                ## load fits header
                hdulist = pyfits.open(self.fitsimage_filename, memmap=True)
                # fix broken cards
                # hdulist[0].verify('silentfix')
                prihdr = hdulist[0].header
            else:
                # doesn't fail if FITS image no present
                print(dfn_utils.warning + " There is no FITS image " + self.fitsimage_filename + ". Cannot generate proper metadata")
                self.tablemetadata = {}
                exit(2)
            
        
        # create output table metadata
        immetaobj = parseObserver.dfn_image_metadata(prihdr, overridegps=True)
        im_meta_dic = immetaobj.toDict()
        
        self.dummy_table = Table()
        self.dummy_table.meta.update(dict(im_meta_dic))
        self.dummy_table.meta['event_codename'] = self.eventcodename
        self.dummy_table.meta['self_disk_path'] = self.output_filename
        self.dummy_table.meta['GFE_standard_version'] = GFE_standard_version
        self.tablemetadata = self.dummy_table.meta
        
                
        # determine encoding firmware
        try:
            self.encoding_firware = timing_corrections.determine_firmware_version(self.dummy_table, table_loc='camera')
        except timing_corrections.UnknownEncodingError as e:
            print(e)
            print("CRITICAL: could not determine shutter encoding. Assuming {}".format('april2014'))
            self.encoding_firware = 'april2014'
        
        print("Encoding firmware detected: " + self.encoding_firware)
        
        # TODO FIXME HACK TERRIBLE
        # extract datetime from image filename
        try:
            unrounded_datetime = datetime.datetime(year=int(self.image_filename[3:7]),
                                                month=int(self.image_filename[8:10]),
                                                day=int(self.image_filename[11:13]),
                                                hour=int(self.image_filename[14:16]),
                                                minute=int(self.image_filename[16:18]),
                                                second=int(self.image_filename[18:20]))
        except ValueError:
            unrounded_datetime = datetime.datetime(year=int(self.image_filename[4:8]),
                                                month=int(self.image_filename[9:11]),
                                                day=int(self.image_filename[12:14]),
                                                hour=int(self.image_filename[15:17]),
                                                minute=int(self.image_filename[17:19]),
                                                second=int(self.image_filename[19:21]))
            

        # round to nearest 30 seconds
        seconds = (unrounded_datetime - unrounded_datetime.min).seconds + (unrounded_datetime - unrounded_datetime.min).days * 86400
        rounded_seconds = ((seconds + 15) // 30) * 30
        self.datetime = unrounded_datetime.min + datetime.timedelta(seconds=rounded_seconds)
        
        # plot the image on the axes
        axcolor = 'gray'
        try:
            axcmap = plt.axes([0.01, 0.4, 0.08, 0.2], axisbg=axcolor)
        except:
            axcmap = plt.axes([0.01, 0.4, 0.08, 0.2], facecolor=axcolor)
        # list of color maps: http://matplotlib.org/examples/color/colormaps_reference.html
        self.radio = RadioButtons(axcmap, ('gray',
                                        'spectral',
                                        'inferno',
                                        'bone',
                                        'hot',
                                        'gnuplot2',
                                        'nipy_spectral',
                                        'gist_stern'), active=0)
        self.radio.on_clicked(cmap_func)
        axvmin = plt.axes([0.25, 0.1, 0.65, 0.03])
        axvmax = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.vmin = Slider(axvmin, 'V_min', np.min(img), np.max(img), valinit=np.min(img))
        self.vmax = Slider(axvmax, 'V_max', np.min(img), np.max(img), valinit=np.max(img))
        self.vmin.on_changed(scales_func)
        self.vmax.on_changed(scales_func)
        self.imgplot = self.ax.imshow(img, cmap='gray', interpolation='none')
        self.imgplot.set_clim(vmin=np.min(img), vmax=np.max(img))
        
        #self.imgplot.set_cmap('spectral')
        #self.fig.canvas.draw()

        # initialise attributes
        self.sequence_index = startSeqIndex
        self.start = 'S'
        self.direction_up = True
        self.text = plt.figtext(0.5, 0.05, '', ha='center')
        self.update_sequence_preview()

        # create list to hold point instances
        self.points = []

        # if previous data is specified, import and draw it
        if previous_points_path:
            self.previous_points_path = previous_points_path
            self.previous_points_directory = os.path.dirname(self.previous_points_path)
            if self.previous_points_directory == '':
                self.previous_points_directory = '.'
            self.previous_points_basename = os.path.basename(self.previous_points_path)
            self.previous_points_filename = os.path.splitext(self.previous_points_basename)[0]

            #self.output_filename = self.previous_points_directory + '/' + '_'.join(self.previous_points_filename.split('_', 5)[:5])

            prev_t = Table.read(self.previous_points_path, format='ascii.ecsv', delimiter=',')
            prev_t['x_image'] = prev_t['x_image'] * scaling
            prev_t['y_image'] = prev_t['y_image'] * scaling

            if 'shutter_inversion' in prev_t.meta.keys():
                if prev_t.meta['shutter_inversion'] in ['true', 'True', 'Y', 'y']:
                    self.shutter_inversion = True
                else:
                    self.shutter_inversion = False
                print('OVERRIDING user input shutter inversion value, using value from loaded point file: ' + str(self.shutter_inversion))

            # make points
            for i in prev_t:
                self.points.append(point(i['x_image'],
                                         i['y_image'],
                                         #datetime.datetime.strptime(i['datetime'], '%Y-%m-%dT%H:%M:%S.%f'),
                                         i['time_err_minus'],
                                         i['time_err_plus'],
                                         int(i['de_bruijn_sequence_element_index']),
                                         i['dash_start_end'],
                                         self.ax.add_artist(plt.Circle((i['x_image'], i['y_image']), 0.2, color=(1, 0, 0, 0.5))), self.encoding_firware))
            print('{0} points imported'.format(len(prev_t)))
            # draw points
            self.fig.canvas.draw()
            
        

        # connect callbacks
        self.cidkey = self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)
        self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        #self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.motion_callback)
        #self.cidmotion = self.fig.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.cidscroll = self.fig.canvas.mpl_connect('scroll_event', self.scroll_callback)
        
        
    def minimum_picking_table(self):
        c1 = Column(name='x_image', dtype=np.dtype(float), unit=u.pix,
                                    data=[i.x for i in self.points])
        c4 = Column(name='y_image', dtype=np.dtype(float), unit=u.pix,
                                    data=[i.y for i in self.points])
        
        
        c10 = Column(name='de_bruijn_sequence_element_index', dtype=np.dtype(int),
                                    data=[i.sequence_index for i in self.points])
        c11 = Column(name='dash_start_end', dtype=np.dtype(str),
                                    data=[i.start for i in self.points])
        
        c8 = Column(name='time_err_minus', dtype=np.dtype(float), unit=u.second,
                                    data=[i.time_err_minus for i in self.points])
        c9 = Column(name='time_err_plus', dtype=np.dtype(float), unit=u.second,
                                    data=[i.time_err_plus for i in self.points])
        
        t = Table([c1, c4, c8, c9, c10, c11], meta=self.tablemetadata)
        timing_corrections.correct_timing(t, self.encoding_firware)
        
        return t


    # keyboard key press callback
    def key_press_callback(self, event):
        # show velocity norm plot in pixel coordinates
        if event.key == 'g':
            
            plt.figure(2)
            
            t = self.minimum_picking_table()
            
            # put points in time order
            t.sort('datetime')
            import test_PP_quality as TPPQ
            TPPQ.show_norm(t)

        # write data to file
        if event.key == 'w':
            # create write time string
            now_timestamp = datetime.datetime.now()
            writetime = now_timestamp.strftime('%Y-%m-%dT%H:%M:%S')
            writetimeshort = now_timestamp.strftime('%Y-%m-%d_%H%M%S')  # 2015-04-17_200358
            # get user name
            user = getpass.getuser()
            # get computer name
            hostname = socket.gethostname()
            # build filename: original file name, user, comment, timestamp
            filename = self.output_filename + '_' + self.eventcodename + '_' + writetimeshort + '_' + user + '_' + self.user_comment + '.ecsv'
            print('Writing {0} point(s) to disk: '.format(len(self.points)) + filename)
            # create columns
            
            t = self.minimum_picking_table()
            
            
            c2 = Column(name='err_minus_x_image', dtype=np.dtype(float), unit=u.pix,
                                      data=[i.x_err_minus for i in self.points])
            c3 = Column(name='err_plus_x_image', dtype=np.dtype(float), unit=u.pix,
                                      data=[i.x_err_plus for i in self.points])
            c5 = Column(name='err_minus_y_image', dtype=np.dtype(float), unit=u.pix,
                                      data=[i.y_err_minus for i in self.points])
            c6 = Column(name='err_plus_y_image', dtype=np.dtype(float), unit=u.pix,
                                      data=[i.y_err_plus for i in self.points])
            c12 = Column(name='periodic_pulse', dtype=np.dtype(str),
                                       data=[i.periodic_pulse for i in self.points])
            c13 = Column(name='pick_flag', dtype=np.dtype(str),
                                       data=[i.pick_flag for i in self.points])
            c14 = Column(name='encoding_type', dtype=np.dtype(str),
                                       data=[i.encoding_type for i in self.points])
            
            t.add_columns([c2, c3, c5, c6, c12, c13, c14])
            
            # create session metadata
            sessionmeta = {'point_picking_write_time': writetime,
                           'point_picking_user': user,
                           'point_picking_hostname': hostname,
                           'point_picking_user_comment': self.user_comment,
                           'point_picking_software': __scriptName__ + " " + __version__,
                           'point_picking_hostname': hostname,
                           'event_codename': self.eventcodename.__str__(),
                           'shutter_inversion': self.shutter_inversion.__str__(),
                           'image_file': self.image_basename}  # FUCKING HELL UNICODE BULLSHIT!!
        
            # merge the metadata dictionary
            t.meta.update(sessionmeta)

            # put points in time order
            t.sort('datetime')
            # write table to disk in ecsv format
            #t.convert_unicode_to_bytestring(python3_only=True)
            
            if not np.all(dfn_utils.has_reliable_timing(t)):
                # ask for confirmation
                print("You have picked some points WITHOUT precise timing information.")
                not_decodable = dfn_utils.query_yes_no("Confirm timing is NOT decodable on this image:", None)
                t.meta['timing_decodable'] = not not_decodable
            
            t.write(filename, format='ascii.ecsv', delimiter=',')
            print(dfn_utils.ok + '{0} point(s) written to disk: '.format(len(self.points)) + filename)

    # mouse button press callback
    def button_press_callback(self, event):
        # if cursor is outside of image area do nothing
        if event.inaxes != self.ax:
            return
        # if pan or box zoom etc. aren't selected
        state = self.fig.canvas.manager.toolbar.mode
        if state == '':
        #if self.fig.canvas.manager.toolbar._active is None:
            # left click
            if event.button == 1:
                self.remove_duplicate_points()
                # add point to list and create matplotlib Circle artist
                self.points.append(point(event.xdata, event.ydata, default_timing_error_minus_s, default_timing_error_plus_s, self.sequence_index, self.start, self.ax.add_artist(plt.Circle((event.xdata, event.ydata), 0.2, color=(1, 0, 0, 0.5))), self.encoding_firware))
                
                # increment or decrement sequence index
                self.increment_decrement_index()
                # draw new point
                self.fig.canvas.draw()
            # right click
            elif event.button == 3:
                # remove closest point
                self.remove_closest_point(event.xdata, event.ydata)
                self.fig.canvas.draw()
            # middle click for points with unknown timing
            elif event.button == 2:
                self.remove_duplicate_points()
                # calculate point datetime
                time = self.sequence_index * 0.1
                # add point to list and create matplotlib Circle artist
                # calculates accurate timing errors
                self.points.append(point(event.xdata, event.ydata, time, 30 - time, self.sequence_index, self.start, self.ax.add_artist(plt.Circle((event.xdata, event.ydata), 0.2, color=(1, 0, 0, 0.5))), self.encoding_firware))
                
                # increment or decrement sequence index
                self.increment_decrement_index()
                # draw new point
                self.fig.canvas.draw()

    # mousewheel scroll callback
    def scroll_callback(self, event):
        # if cursor is outside of image area do nothing
        if event.inaxes != self.ax:
            return
        # if pan or box zoom etc. aren't selected
        state = self.fig.canvas.manager.toolbar.mode
        if state == '':
            if event.button == 'up':
                self.direction_up = True
                self.increment_decrement_index()
            if event.button == 'down':
                self.direction_up = False
                self.increment_decrement_index()

    # def motion_callback(self, event):
        # pass

    # def button_release_callback(self, event):
        # pass

    def remove_duplicate_points(self):
        # find indices of all points with the same index and start as current
        duplicate_indices = [i for i, x in enumerate(self.points) if x.sequence_index == self.sequence_index and x.start == self.start]
        # remove the corresponding Circle artists
        [self.points[i].circle_instance.remove() for i in duplicate_indices]
        # delete the point instances
        for i in duplicate_indices:
            del self.points[i]

    def remove_closest_point(self, x, y):
        # find the index of the closest point
        point_index = np.argmin([sqrt((i.x - x)**2 + (i.y - y)**2) for i in self.points])
        # remove the corresponding Circle artist
        self.points[point_index].circle_instance.remove()
        # update the element index and start to the same as the point that was just removed to make correcting picked points easy
        self.sequence_index = self.points[point_index].sequence_index
        self.start = self.points[point_index].start
        self.update_sequence_preview()
        # delete the point instance
        del self.points[point_index]

    def up_down_text(self):
        if self.direction_up == True:
            return_text = '+t'
        else:
            return_text = '-t'
        return return_text

    # updates the sequence index info and the sequence preview
    def update_sequence_preview(self):
        self.text.set_text('{0:0>3} {1} {2}\n{3}'.format(self.sequence_index, self.start, self.up_down_text(),
                                                         ''.join(map(str, (db_seq * 3)[self.sequence_index - 20 + len(db_seq):self.sequence_index + len(db_seq)] +
                                                                     ['.'] + [db_seq[self.sequence_index]] + ['.'] +
                                                                     (db_seq * 3)[self.sequence_index + 1 + len(db_seq):self.sequence_index + 21 + len(db_seq)]))))
        self.fig.canvas.draw()

    # increments or decrements the index and start/end based on the current direction
    def increment_decrement_index(self):
        # picking a rocket or something
        if self.pick_start_and_end_exposure:
            if self.sequence_index == 0:
                # move to end
                self.sequence_index = int(float(self.tablemetadata['exposure_time']) * 10)
            else:
                # move back to 0
                self.sequence_index = 0
        else:
            
            if self.direction_up == True:
                if self.encoding_firware == 'oukaimeden_PW_ends_only':
                    self.sequence_index = self.sequence_index + 1
                elif self.start == 'S':
                    if self.encoding_firware == 'april2014' or self.encoding_firware == 'boogardie_PW_1_0_continuous' or db_seq[self.sequence_index] == 1:
                        self.start = 'E'
                    # if using new dot encoding, skip the 0-end element because it does not exist
                    else:
                        self.sequence_index = self.sequence_index + 1
                else:
                    self.sequence_index = self.sequence_index + 1
                    self.start = 'S'
            else:
                if self.encoding_firware == 'oukaimeden_PW_ends_only':
                    self.sequence_index = self.sequence_index - 1
                elif self.start == 'S':
                    if self.encoding_firware == 'april2014' or self.encoding_firware == 'boogardie_PW_1_0_continuous' or db_seq[self.sequence_index-1] == 1:
                        self.start = 'E'
                        self.sequence_index = self.sequence_index - 1
                    # if using new dot encoding, skip the 0-end element because it does not exist
                    else:
                        self.start = 'S'
                        self.sequence_index = self.sequence_index - 1
                else:
                    self.start = 'S'
                    
        self.update_sequence_preview()
        self.fig.canvas.draw()


def main():
    # command line argument parser
    parser = argparse.ArgumentParser(description='Semi-automated point picker that finds weighted centroids')
    parser.add_argument("-i", "--image_path", type=str, help="Path to fireball image", required=True)
    parser.add_argument("-e", "--previous_points_path", type=str, default=None, help='Previously picked ecsv.')
    parser.add_argument("-c", "--comment", type=str, default="nocomment", help="Comment. Purpose of the point picking (rough, very precise, fragment number)")
    parser.add_argument("-f", "--fireball", type=str, default="fireball", help="Fireball codename (eg. DN150621)")
    parser.add_argument("-r", "--reverse_DB_Seq", action="store_true", default=False,
                        help="Inverts DB seq: 1s become 0s, 0s become 1s")
    parser.add_argument("-s", "--scaling", type=float, default=1.0,
                        help="Scale the imported points. Ex: you have points picked on a 6x4 pixels image. If you want to work on the dowsampled version 3x2 pixels, use 0.5 scaling factor.")
    parser.add_argument("-n", "--startSeqIndex", type=str, default="0",
                        help="Start index: integer, or auto (finds nearby files)")
    parser.add_argument("-I", "--shutter_inversion", action="store_true", default=False,
                        help="Shutter Inversion (ends are starts, and starts are ends)")
    parser.add_argument("-X", "--startandend", action="store_true", default=False,
                        help="Pick only start and end of exposure")

    args = parser.parse_args()
    
    global db_seq
    
    if args.startSeqIndex == 'auto':
        seq_decoded = False
        startSeqIndex = 0
        event_folder = os.path.dirname(os.path.dirname(args.image_path))
        print("Previous point picking for that event:")
        for dirName, _, fileList in os.walk(event_folder):
        # skip if folder is a trajectory folder, nearby folder, fragment folder, or some other irrelevant shit
            if any(s in dirName for s in ['trajectory_', 'nearby', 'fragment',
                                        'old', 'IDLtrajFolderName', 'misc',
                                        'darkflight', 'trajector', 'WRF', 'search', 'video', '3d']):
                continue
            for i in fileList:
                if i.endswith('.ecsv'):
                    try:
                        t = Table.read(os.path.join(event_folder, dirName, i), format='ascii.ecsv')
                        if np.all(dfn_utils.has_reliable_timing(t)):
                            #print(t['de_bruijn_sequence_element_index'].astype(np.int32).dtype)
                            print("{0:4} points: {1:>3} -> {2:10} {3}".format(len(t), np.min(t['de_bruijn_sequence_element_index'].astype(np.int32)), np.max(t['de_bruijn_sequence_element_index'].astype(np.int32)),i))
                            seq_decoded = True
                    except Exception:
                        pass
        # TODO REMOVE
        #startSeqIndex = int(input('Choose sequence start:'))
        if not seq_decoded:
            print('Cannot find any previous point picking')
            
        open_image = input('Open image? [y]/n') or 'y'
        if open_image == 'y':
            #os.system('{} {} &'.format(IMAGE_VIEWER, args.image_path))
            # if NEF
            if args.image_path.endswith(technical_constants_DFN.DEFAULT_RAW_EXTENSION):
                nef = args.image_path
                thumb_jpg = nef.replace(technical_constants_DFN.DEFAULT_RAW_EXTENSION, 'thumb.jpg')
            # else, probably FITS or TIFF
            else:
                thumb_jpg = args.image_path[:-7]+'.thumb.jpg'
                nef = args.image_path.split('_')[-1]
            if not os.path.isfile(thumb_jpg):
                os.system(f'dcraw -e {nef}') 
            if not os.path.isfile(thumb_jpg):
                print(f'cannot find {thumb_jpg} or a way to generate it from raw')
            else:
                os.system(f'{IMAGE_VIEWER} {thumb_jpg} &')
        
        db_string_full = "".join([str(e) for e in db_seq])
        #1print(db_string_full)
        search_sequence = input('Search sequence? [y]/n') or 'y'
        while search_sequence == 'y':
            user_decoded_sequence = input('Read sequence in image:')
            if len(user_decoded_sequence) < 9:
                print('WARNING, entered sequence is less than 9 elements long, uniqueness is not guaranteed.')
            # TODO IMPLEMENT PROPERLY ROB
            found_index = db_string_full.find(user_decoded_sequence)
            if found_index != -1:
                print('Found index in sequence: {}'.format(found_index))
                startSeqIndex = found_index
                break  
            else:
                print()
            search_sequence = input('Search sequence again? [y]/n') or 'y'
        else:
            startSeqIndex = int(input('Choose sequence start:'))
            
            
            
    else:
        startSeqIndex = int(args.startSeqIndex)

    if args.reverse_DB_Seq:
        db_seq = np.abs(db_seq - np.ones(len(db_seq), dtype=np.int8)).tolist()
        print("DB sequence inverted:", db_seq)

    pointPickingComment = args.comment.translate({ord(c): "" for c in "!@#$%^&*()[]{};:,./<>?\|`~=+"})
    pointPickingComment = pointPickingComment.translate({ord(c): "-" for c in " "})
    fireballName = args.fireball

    picker_instance = picker(args.image_path,
               args.previous_points_path,
               pointPickingComment,
               fireballName,
               args.scaling,
               startSeqIndex=startSeqIndex,
               shutter_inversion=args.shutter_inversion,
               pick_start_and_end_exposure=args.startandend)
    try:
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
    except Exception as e:
        print(e)
        pass
    plt.show()


if __name__ == '__main__':
    main()
