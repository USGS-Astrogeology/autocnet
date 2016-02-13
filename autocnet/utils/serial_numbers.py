import glob
import datetime
import pvl
import time

from pvl.decoder import PVLDecoder

from autocnet.examples import get_path
from autocnet.fileio import io_json
from autocnet.utils import utils

# See README in the translation file directory for information on this dict
SN_TRANSLATION_FILE_LOOKUP = io_json.read_json(get_path('sn_translation_file_lookup.json'))


def lookup_sn_translation_file(cube_obj):
    """
    This function attempts to find the appropriate default serial number translation file for
    the given IsisCube PVL object by searching for a matching (SpacecraftName, InstrumentId)
    keyword value combination from the object's Instrument group. 

    Parameters
    ----------
    cube_obj : PvlModule object
               The IsisCube PVL object found in the cube's label.

    Returns
    -------
     : str
       The full file name (with absolute path) for the default translation table.

    Raises
    ------
    NotImplementedError
       This error is raised when a matching (SpacecraftName, InstrumentId) combination is not
       found in the JSON look up file.
    """
    try:
        inst_group = utils.find_in_dict(cube_obj, 'Instrument')
        sc_name    = inst_group['SpacecraftName']
        inst_id    = inst_group['InstrumentId']
        spacecraft = sc_name.lower().strip().replace("_","").replace(" ","")
        instrument = inst_id.lower().strip().replace("_","").replace(" ","")
        return get_path(SN_TRANSLATION_FILE_LOOKUP[spacecraft][instrument])
    except:
        raise NotImplementedError("Unable to find matching (SpacecraftName, InstrumentId) in "
                                  "the translation file lookup dict. This cube requires a "
                                  "translation_file to be provided.")


class SerialNumberDecoder(PVLDecoder):
    """
    A PVL Decoder class to handle cube label parsing for the purpose of creating a valid ISIS
    serial number. Inherits from the PVLDecoder in planetarypy's pvl module. 
    """
    def cast_unquoated_string(self, value):
        """
        Overrides the parent class's method so that any un-quoted string type value found in the
        parsed pvl will just return the original value. This is needed so that keyword values
        are not re-formatted from what is originally in the ISIS cube label.

        Note: This affects value types that are recognized as null, boolean, number, datetime, 
        et at.
        """
        return value.decode('utf-8')


def get_serial_number(label_file, translation_file='None'):
    """
    Retrieves the serial number for the given ISIS cube.

    Parameters
    ----------
    label_file : str
                Full path to the file containing a valid ISIS cube label. 
                This may be a cube or detached label.

    translation_file: str
                      Name of the ISIS serial number translation file to be used. If 'None' is 
                      given, the function will attempt to find a default translation file using
                      the SpacecraftName and InstrumentId found in the cube label. 
                      Default is 'None'.

    Returns
    -------
     : str
       The ISIS compliant serial number associated with the given cube.


    Raises
    ------
    ValueError
       This error occurs when the translation file has keywords that don't exist in the 
       specified location (InputGroup) of the given ISIS cube.
    """

    label = pvl.load(label_file, cls=SerialNumberDecoder)
    cube_obj = utils.find_in_dict(label, 'IsisCube')

    if translation_file == 'None':
        translation_file = lookup_sn_translation_file(cube_obj)

    translation = pvl.load(translation_file)
    sn = []
    for sn_group in translation:
        input_group = utils.find_in_dict(sn_group[1], 'InputGroup')
        group_location = input_group.split(',')
        label_group = utils.find_in_dict(cube_obj, group_location[-1])
        input_key = utils.find_in_dict(sn_group[1], 'InputKey')
        label_value = utils.find_in_dict(label_group, input_key)
        if not isinstance(label_value, str):
            raise ValueError("Invalid ISIS cube label or translation file. Serial number "
                             "translation file keyword [" + input_key + "] was not found in "
                             "the ISIS cube label.")
             
        sn_keyword_name = sn_group[0]
        keywords = sn_group[1]
        for keyword in keywords:
            if keyword[0] == 'Translation':
                if keyword[1][1] == label_value or keyword[1][1] == '*':
                    if keyword[1][0] == '*':
                        sn.append(label_value)
                    else:
                        sn.append(keyword[1][0])
                    break
    return '/'.join(sn)
