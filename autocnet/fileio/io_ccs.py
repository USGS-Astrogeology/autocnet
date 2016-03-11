# This code is used to read individual ChemCam CCS .csv files
# Header data is stored as attributes of the data frame
# White space is stripped from the column names
import os
import numpy as np
import pandas as pd
import scipy
from autocnet.fileio.header_parser import header_parser
from autocnet.fileio.utils import file_search


def CCS(input_data):
    df = pd.DataFrame.from_csv(input_data, header=14)
    # strip whitespace from column names
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df = df.transpose()

    cols = df.columns.tolist()
    for i, x in enumerate(cols):
        cols[i] = ('wvl', round(float(x), 5))
    df.columns = pd.MultiIndex.from_tuples(cols)
    # extract info from the file name
    fname = os.path.basename(input_data)
    df['sclock'] = fname[4:13]
    df['sclock'] = pd.to_numeric(df['sclock'])
    df['seqid'] = fname[25:34].upper()
    df['Pversion'] = fname[34:36]
    # transpose the data frame

    # read the file header and put information into the dataframe as new columns
    # (inefficient to store this data many times, but much easier to concatenate data from multiple files)
    with open(input_data, 'r') as f:
        header = {}
        for i, row in enumerate(f.readlines()):
            if i < 14:
                row = row.split(',')[0]
                header.update(header_parser(row, '='))

    for label, data in header.items():
        if '_float' in label:
            label = label.replace('_float', '')
        if label == 'dark':
            label = 'darkspec'
        df[label] = data

    df.index.rename('shotnum', inplace=True)
    df.reset_index(level=0, inplace=True)
    return df


def CCS_SAV(input_data):

    d = scipy.io.readsav(input_data, python_dict=True)
    # combine the three spectrometers
    spectra = np.vstack([d['uv'], d['vis'], d['vnir']])
    aspectra = np.array([np.hstack([d['auv'], d['avis'], d['avnir']])]).T
    mspectra = np.array([np.hstack([d['muv'], d['mvis'], d['mvnir']])]).T

    # create tuples for the spectral columns to use as multiindex
    wvls = list(np.hstack([d['defuv'], d['defvis'], d['defvnir']]))
    for i, x in enumerate(wvls):
        wvls[i] = ('wvl', round(x, 5))

    # define column names
    shotnums = list(range(1, d['nshots'] + 1))
    shots = ['shot' + str(i) for i in shotnums]
    shots.extend(['ave', 'median'])

    # create the data frame to hold the spectral data
    df = pd.DataFrame(np.hstack([spectra, aspectra, mspectra]),
                      columns=shots, index=pd.MultiIndex.from_tuples(wvls))
    df = df.transpose()

    # remove the above elements from the dict
    to_remove = [
        'uv',
        'vis',
        'vnir',
        'auv',
        'avis',
        'avnir',
        'muv',
        'mvis',
        'mvnir',
        'defuv',
        'defvis',
        'defvnir',
        'label_info']
    for x in to_remove:
        del d[x]

    # extract info from the file name
    fname = os.path.basename(input_data)
    d['sclock'] = fname[4:13]
    d['seqid'] = fname[25:34].upper()
    d['Pversion'] = fname[34:36]

    # Add metadata to the data frame by stepping through the dict
    for label, data in d.items():
        if isinstance(data, bytes):
            data = data.decode()
        df[label] = data

    df['sclock'] = pd.to_numeric(df['sclock'])
    df.index.rename('shotnum', inplace=True)
    df.reset_index(level=0, inplace=True)

    return df


def ccs_batch(directory, searchstring='*CCS*.csv', is_sav=False):

    if 'SAV' in searchstring:
        is_sav = True
    else:
        is_sav = False
    filelist = file_search(directory, searchstring)
    basenames = np.zeros_like(filelist)
    sclocks = np.zeros_like(filelist)
    P_version = np.zeros_like(filelist, dtype='int')

    # Extract the sclock and version for each file and ensure that only one
    # file per sclock is being read, and that it is the one with the highest
    # version number
    for i, name in enumerate(filelist):
        basenames[i] = os.path.basename(name)
        sclocks[i] = basenames[i][4:13]
        P_version[i] = basenames[i][-5:-4]
    sclocks_unique = np.unique(sclocks)
    filelist_new = np.array([], dtype='str')
    for i in sclocks_unique:
        match = (sclocks == i)
        maxP = P_version[match] == max(P_version[match])
        filelist_new = np.append(filelist_new, filelist[match][maxP])

    filelist = filelist_new
    # any way to speed this up for large numbers of files?
    # Should add a progress bar for importing large numbers of files
    for i in filelist:
        if is_sav:
            tmp = CCS_SAV(i)

        else:
            tmp = CCS(i)

        try:
            # This ensures that rounding errors are not causing mismatches in
            # columns
            cols1 = list(combined['wvl'].columns)
            cols2 = list(tmp['wvl'].columns)
            if set(cols1) == set(cols2):
                combined = pd.concat([combined, tmp])
            else:
                print("Wavelengths don't match!")
        except:
            combined = tmp
    return combined
