#!/usr/bin/env python
#
# Python 2 and 3 compatible as long as astropy remains 2/3 compatible


__author__ = "Hadrien Devillepoix"
__copyright__ = "Copyright 2017, Desert Fireball Network"
__license__ = "MIT"
__version__ = "1.1"
__scriptName__ = "timing_corrections.py"



# general
import logging
import os
# science
import numpy as np
from astropy.time import Time, TimeDelta

# local
import dfn_utils
import de_bruijn


MANUAL_OVERRIDE_LIST = [
    {'telescope': 'DFNKIT11', 'start_date': Time('2018-09-18'), 'end_date': Time('2019-01-31'), 'firmware': 'april2014'},
    {'telescope': 'DFNSMALL58', 'start_date': Time('2018-10-12'), 'end_date': Time('2018-12-04'), 'firmware': 'april2014'},
    {'telescope': 'DFNEXT029', 'start_date': Time('2019-01-01'), 'end_date': Time('2019-06-01'), 'firmware': 'boogardie_PW_1_0_continuous'},
    {'telescope': 'DFNSMALL12', 'start_date': Time('2020-07-05'), 'end_date': Time('2020-10-17'), 'firmware': 'boogardie_PW_1_0_continuous'},
    {'telescope': 'DFNSMALL45', 'start_date': Time('2020-05-10'), 'end_date': Time('2021-06-01'), 'firmware': 'oukaimeden_PW_ends_only'}
    ]


class UnknownEncodingError(Exception):
    '''
    Exception class used when data time encoding method in unknown
    '''
    pass


def check_manual_fw_override(table, manual_override_list=MANUAL_OVERRIDE_LIST):
    # fast way to get out
    if not table.meta['telescope'] in [e['telescope'] for e in manual_override_list]:
        return None
    
    # actual thourough check
    for e in manual_override_list:
        if not table.meta['telescope'].startswith(e['telescope']):
            continue
        print('Camera {} is on manual override list, checking dates. {}'.format(table.meta['telescope'], e))
        obs_time = Time(table.meta['isodate_start_obs'])
        if obs_time > e['start_date'] and obs_time < e['end_date']:
            print('Table is on manual firmware override list')
            return e['firmware']
    else:
        return None
    


def shift_time(table, delta):
    
    td = TimeDelta(delta, format='sec')
    table['datetime'] = (Time(table['datetime']) + td).isot



def get_uC_firmware_version_from_log(log_file):
    '''
    
    '''
    logger = logging.getLogger('trajectory')
    
    fw_string = dfn_utils.search_dfn_operation_log(log_file, key='leostick_version', module='interval_control_lin')
    
    #print(fw_string)
    
    if ("uilt: 10:56:18 Apr  1 2014" in fw_string or
       "camera triggering fixed : 10:48:31 Oct 21 2014" in fw_string):
        fw = 'april2014'
    elif ("uilt:May 30 2017 16:37:45 note:kit" in fw_string or
          "uilt:May 26 2017 10:51:00 note:kit, small and ext compatible" in fw_string or
          "uilt:Jul 23 2015 17:33:48 note:pulse frequency encoding, bulb mode" in fw_string):
        fw = 'june2017'
    else:
        raise UnknownEncodingError()
    
    logger.info('Determined the firmware version using the interval log. FW: {}'.format(fw))
    
    return fw
    
    
def correct_timing_smart(table):
    '''
    Correct points timing
    '''
    logger = logging.getLogger('trajectory')
    
    if not 'origin' in table.meta:
        logger.error('Unknown data origin, cannot correct de Bruijn timing')
        return
    if 'Desert Fireball Network' not in table.meta['origin']:
        logger.warning('Astrometric data not from DFN ({}), timing correction not implemented'.format(table.meta['origin']))
        return
    
    
    # find out what encoding was used
    try:
        firmware = determine_firmware_version(table)
    except UnknownEncodingError as err:
        logger.error(err)
        logger.critical('Skipping timing correction for {0}'.format(table.meta['telescope']))
        return
    
    # correct timing
    
    correct_timing(table, firmware)
    


def determine_firmware_version(table, table_loc='trajectory'):
    '''
    
        table_loc: which folder self_disk_path points to.
            'trajectory' (default): trajectory folder (need to a bit of path navigation in this case)
            'camera': camera folder
    '''
    logger = logging.getLogger('trajectory')
    
    telescope = table.meta['telescope']
    
    # try reading interval log first
    try:
        if table_loc == 'trajectory':
            event_dir = os.path.dirname(os.path.realpath(os.path.dirname(table.meta['self_disk_path']))) # UNSAFE FIXME
        elif table_loc == 'camera':
            event_dir = os.path.dirname(table.meta['self_disk_path'])
        else:
            raise UnknownEncodingError("cannot find camera base directory")
        
        # if for some reason that camera at that particular time was not running the expected FW
        manual_override = check_manual_fw_override(table)
        if manual_override:
            logger.info('camera firmware corresponds to entry in manual firmware override list. Using ovveride found: {}'.format(manual_override))
            return manual_override
        
        interval_log_file = dfn_utils.find_log_file(event_dir, suffix='_log_interval', extension='txt', system_number=telescope)
        return get_uC_firmware_version_from_log(interval_log_file)
    except Exception as err:
        logger.error(err)
        pass
    
    # default to best assumption
    sequence_reset_time = dfn_utils.round_to_nearest_30_seconds(table.meta['isodate_start_obs'])
    if ('DFNEXT' in telescope or
        'DFNKIT' in telescope or
        ('DFNSMALL' in telescope and (sequence_reset_time > Time('2017-06-01') and sequence_reset_time < Time('2017-09-15')))):
        fw = 'june2017'
        logger.warning('Determined the firmware version using dodgy assumptions. FW: {}'.format(fw))
        return fw
    
    elif 'DFNSMALL' in telescope:
        logger.warning('Determined the firmware version using dodgy assumptions')
        fw = 'april2014'
        logger.warning('Determined the firmware version using dodgy assumptions. FW: {}'.format(fw))
        return fw
    else:
        raise UnknownEncodingError()


def correct_timing(table, firmware):
    '''
    Correct points timing based on firmware version used
    '''
    
    # de Bruijn sequence
    db_seq = de_bruijn.de_bruijn_sequence(2, 9).de_bruijn_sequence
    
    # frequency (Hz)
    frequency = 10.
    period = 1 / frequency
    
    
    table['zero_or_one'] = [db_seq[int(i)] for i in table['de_bruijn_sequence_element_index']]
    
    is_end_binary_int = np.asarray(table['dash_start_end'] == 'E', dtype='int')
    
    
    sequence_reset_time = dfn_utils.round_to_nearest_30_seconds(table.meta['isodate_start_obs'])
    # HACK
    if 'event_codename' in table.meta and table.meta['event_codename'] == 'DN170607_01':
        sequence_reset_time = Time('2017-06-07T15:01:30')
    
    
    telescope = table.meta['telescope']
    
    
        
    sequence_position_shift = TimeDelta(np.asarray(table['de_bruijn_sequence_element_index'], dtype='int') * period,
                                            format='sec')

        
    if firmware == 'june2017':
        offset = TimeDelta(0., format='sec')
        element_shift = TimeDelta(0.05 * is_end_binary_int,
                            format='sec')
        table.meta['timing_de_bruijn_encoding_type'] = 'pulse_frequency'
    
    elif firmware == 'april2014':
        offset = TimeDelta(1.02637,
                           format='sec')
        element_shift = TimeDelta((0.02 + table['zero_or_one'] * 0.04) * is_end_binary_int,
                            format='sec')
        table.meta['timing_de_bruijn_encoding_type'] = 'pulse_width'
    
    elif firmware == 'boogardie_PW_1_0_continuous' or firmware == 'oukaimeden_PW_ends_only':
        offset = TimeDelta(0., format='sec')
        element_shift = TimeDelta(0.05 * is_end_binary_int,
                            format='sec')
        table.meta['timing_de_bruijn_encoding_type'] = 'pulse_width_1_0_continuous'
        
    else:
        raise UnknownEncodingError()
    


    
    # not accurate
    if ('DFNSMALL07' in telescope or
        ('DFNSMALL10' in telescope and sequence_reset_time < Time('2016-01-01'))):
        no_pps_shift = TimeDelta(0.4,
                           format='sec')
    else:
        no_pps_shift = TimeDelta(0.,
                           format='sec')
    
    
    table['datetime'] = (sequence_reset_time +
                         sequence_position_shift +
                         offset +
                         element_shift +
                         no_pps_shift).isot
    
    table.meta['timing_corrected'] = True
    table.meta['timing_correction_dictionary_version'] = 'DEBUG'
    
    #print('Timing has been corrected using firware version information.')
    
    
        
