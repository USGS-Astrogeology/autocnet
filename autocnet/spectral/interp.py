# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:25:15 2015



@author: rbanderson
"""
import numpy as np
import pandas as pd
import scipy as sp


def interp_spect(old_df, xnew):
    xnew = np.array(xnew, dtype='float')

    metadata_cols = old_df.columns.levels[0] != 'wvl'
    metadata = old_df[old_df.columns.levels[0][metadata_cols]]
    old_wvls = np.array(old_df['wvl'].columns, dtype='float')
    old_spectra = np.array(old_df['wvl'])
    new_spectra = np.empty([len(old_spectra[:, 0]), len(xnew)]) * np.nan
    interp_index = (xnew > min(old_wvls)) & (xnew < max(old_wvls))

    f = sp.interpolate.interp1d(old_wvls, old_spectra, axis=1)
    new_spectra[:, interp_index] = f(xnew[interp_index])
#    plot.plot(old_wvls,old_spectra[0,:])
#    plot.plot(xnew,new_spectra[0,:])
#    plot.ylim([0,1e11])
#    plot.xlim([290,300])
    xnew = list(xnew)
    for i, x in enumerate(xnew):
        xnew[i] = ('wvl', x)

    new_df = pd.DataFrame(
        new_spectra,
        columns=pd.MultiIndex.from_tuples(xnew),
        index=old_df.index)
    new_df = pd.concat([new_df, metadata], axis=1)

    return new_df
