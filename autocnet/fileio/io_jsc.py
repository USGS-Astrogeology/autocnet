import os
import numpy as np
import pandas as pd
from autocnet.fileio.utils import file_search
# This function reads the lookup tables used to expand metadata from the file names
# This is separated from parsing the filenames so that for large lists of files the
# lookup tables don't need to be read over and over
#
# Info in the tables is stored in a dict of dataframes so that only one variable
# (the dict) needs to be passed between functions


def read_refdata(LUT_files):
    spectrometer_info = pd.read_csv(LUT_files['spect'], index_col=0)
    # spectrometer_info.reset_index(inplace=True)
    laser_info = pd.read_csv(LUT_files['laser'], index_col=0)
    # laser_info.reset_index(inplace=True)
    exp_info = pd.read_csv(LUT_files['exp'], index_col=0)
    # exp_info.reset_index(inplace=True)
    sample_info = pd.read_csv(LUT_files['sample'], index_col=0)
    # sample_info.reset_index(inplace=True)
    refdata = {
        'spect': spectrometer_info,
        'laser': laser_info,
        'exp': exp_info,
        'sample': sample_info}
    return refdata

# This function parses the file names to record metadata related to the
# observation


def jsc_filename_parse(filename, refdata):
    # strip the path off of the file name
    filename = os.path.basename(filename)
    filename = filename.split('_')  # split the file name on underscores
    libs_ID = filename[0]
    laserID = filename[4][0]
    expID = filename[5]
    spectID = filename[6]
    if libs_ID in refdata['sample'].index:
        file_info = pd.DataFrame(refdata['sample'].loc[libs_ID]).T
    else:
        file_info = pd.DataFrame(refdata['sample'].loc['Unknown']).T
    file_info.index.name = 'LIBS ID'
    file_info.reset_index(level=0, inplace=True)
    file_info['loc'] = int(filename[1])
    file_info['lab'] = filename[2]
    file_info['gas'] = filename[3][0]
    file_info['pressure'] = float(filename[3][1:])

    if laserID in refdata['laser'].index:
        laser_info = pd.DataFrame(refdata['laser'].loc[laserID]).T
        laser_info.index.name = 'Laser Identifier'
        laser_info.reset_index(level=0, inplace=True)
        file_info = pd.concat([file_info, laser_info], axis=1)

    file_info['laser_power'] = float(filename[4][1:])
    if expID in refdata['exp'].index:
        exp_info = pd.DataFrame(refdata['exp'].loc[expID]).T
        exp_info.index.name = 'Exp Identifier'
        exp_info.reset_index(level=0, inplace=True)
        file_info = pd.concat([file_info, exp_info], axis=1)

#    file_info['spectrometer']=spectID
#    if spectID in refdata['spect'].index:
#        temp=refdata['spect'].loc[spectID]
#        temp=[temp[2],temp[4:]]
#        spect_info=pd.DataFrame(refdata['spect'].loc[spectID]).T
#        spect_info.index.name='Spectrometer Identifier'
#        spect_info.reset_index(level=0,inplace=True)
#        file_info=pd.concat([file_info,spect_info],axis=1)

    return file_info


def JSC(input_file, refdata):
    data = pd.read_csv(input_file, skiprows=14, sep='\t')
    data = data.rename(
        columns={
            data.columns[0]: 'time1',
            data.columns[1]: 'time2'})
    # split the two time columns from the data frame
    times = data[['time1', 'time2']]
    # trim the data frame so it is just the spectra
    data = data[data.columns[2:]]

    # make a multiindex for each wavlength column so they can be easily
    # isolated from metadata later
    cols = data.columns.tolist()
    for i, x in enumerate(cols):
        cols[i] = ('wvl', round(float(x), 5))
    data.columns = pd.MultiIndex.from_tuples(cols)

    # create a metadata frame and add the times to it
    metadata = pd.concat(
        [jsc_filename_parse(input_file, refdata)] * len(data.index))
    metadata.index = data.index
    metadata = pd.concat([metadata, times], axis=1)

    # add the metadata columns to the data frame
    for col in metadata.columns.tolist():
        data[col] = metadata[col]

    return data


def jsc_batch(directory, LUT_files, searchstring='*.txt'):
    # Read in the lookup tables to expand filename metadata
    refdata = read_refdata(LUT_files)
    # get the list of files that match the search string in the given directory
    filelist = file_search(directory, searchstring)
    spectIDs = []  # create an empty list to hold the spectrometer IDs

    for file in filelist:
        # get the spectrometer IDs for each file in the list
        spectIDs.append(os.path.basename(file).split('_')[6])
    spectIDs_unique = np.unique(spectIDs)  # get the unique spectrometer IDs
    dfs = []  # create an empty list to hold the data frames for each spectrometer

    # loop through each spectrometer, read the spectra and combine them into a
    # single data frame for that spectrometer
    for spect in spectIDs_unique:
        sublist = filelist[np.in1d(spectIDs, spect)]
        temp = [JSC(sublist[0], refdata)]
        for file in sublist[1:]:
            temp.append(JSC(file, refdata))
        dfs.append(pd.concat(temp))

    # now combine the data frames for the different spectrometers into a
    # single data frame containing all the data
    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.merge(df)

    return combined
