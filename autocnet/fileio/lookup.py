# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 09:51:51 2015

@author: rbanderson

This function uses the pandas merge ability to look up metadata for an existing dataframe in a csv file
If lookupfile is a list, then each file will be read and concatenated together
The default settings are for looking up ChemCam CCS csv data in the ChemCam master list files, matching on sclock value
"""
import pandas as pd


def lookup(
        df,
        lookupfile=None,
        lookupdf=None,
        sep=',',
        skiprows=1,
        left_on='sclock',
        right_on='Spacecraft Clock'):
    if lookupfile is not None:
        # this loop concatenates together multiple lookup files if provided
        # (mostly to handle the three different master lists for chemcam)
        for x in lookupfile:
            try:
                tmp = pd.read_csv(
                    x, sep=sep, skiprows=skiprows, error_bad_lines=False)
                lookupdf = pd.concat([lookupdf, tmp])
            except:
                lookupdf = pd.read_csv(
                    x, sep=sep, skiprows=skiprows, error_bad_lines=False)

    temp = pd.DataFrame(df[left_on])
    metadata = pd.merge(
        temp,
        lookupdf,
        left_on=left_on,
        right_on=right_on,
        how='inner')

    # should add a check here to make sure the indices match up: can cause
    # weird behavior
    for col in metadata.columns:
        df[col] = metadata[col]
    return df
