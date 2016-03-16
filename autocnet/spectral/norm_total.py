# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:20:38 2016

@author: rbanderson
"""
import numpy as np
import pandas as pd


def norm_total(df):
    df = df.div(df.sum(axis=1), axis=0)
    return df


def norm_spect(df, ranges):
    # this is all a mess, trying to work around pandas idiosyncracies. should
    # be re-written
    df_spect = df['wvl']
    wvls = df_spect.columns.values
    df_spect_sub = []
    allind = []
    for i in ranges:
        ind = (wvls >= i[0]) & (wvls <= i[1])
        cols = wvls[ind]
        allind.append(ind)
        df_spect_sub.append(df_spect[cols])
    for i, j in enumerate(df_spect_sub):
        df_spect_sub[i] = norm_total(j)

#        try:
#            df_spect_combined=pd.concat([df_spect_combined,df_spect_sub[i]],ignore_index=True)
#        except:
#            df_spect_combined=df_spect_sub[i]
    allind = np.sum(allind, axis=0)
    wvls_excluded = wvls[np.where(allind != 1)]
    df_excluded = df_spect[wvls_excluded]
    df_spect_combined = pd.concat(df_spect_sub, axis=1)

    df['wvl'] = pd.concat([df_excluded, df_spect_combined], axis=1)

    return df
