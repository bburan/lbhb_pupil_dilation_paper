from hashlib import md5
import pickle

import numpy as np
import pandas as pd
import pystan


def CachedStanModel(model_file, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    with open(model_file, 'rb') as fh:
        model_code = fh.read().decode('utf8')
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached models/cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached models/cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    return sm


def renamer(x):
    if '_' in x:
        return x.split('_')[1]
    return x


def load_rates():
    # NOTE: Pupil should be 0 (small) or 1 (large)
    # Frequency should be octave-spaced

    # Load FTC
    ftc = pd.read_csv('frequency_tuning_curves_for_bburan.csv')
    ftc.columns = [s.replace(' ', '') for s in ftc.columns]
    cols = ['pupil', 'frequency', 'ftc_count', 'ftc_time', 'spont_count', 'spont_time']
    ftc = pd.wide_to_long(ftc, cols, 'cellid', 'idx', sep='_').dropna()
    ftc['pupil'] -= 1
    ftc['frequency'] = np.log2(ftc['frequency'])

    # Load RLF
    rlf = pd.read_csv('rate_level_functions_for_bburan.csv')
    rlf.columns = [s.replace(' ', '') for s in rlf.columns]
    cols = ['pupil', 'level', 'rlf_count', 'rlf_time', 'spont_count', 'spont_time']
    rlf = pd.wide_to_long(rlf, cols, 'cellid', 'idx', sep='_').dropna()
    rlf['pupil'] -= 1

    rlf = rlf.reset_index().set_index(['cellid', 'pupil', 'level'], verify_integrity=True)[['rlf_count', 'rlf_time']].sort_index()

    # Get SR from FTC
    sr = ftc.groupby(['cellid', 'pupil'])[['spont_count', 'spont_time']].first().sort_index()

    ftc = ftc.reset_index().set_index(['cellid', 'pupil', 'frequency'])[['ftc_count', 'ftc_time']].sort_index()
    m = ftc['ftc_time'] > 0
    ftc = ftc.loc[m]

    sig_cells = pd.read_csv('psth_sig_cellids.csv')
    sig_cells['significant'] = True
    sig_cells = sig_cells.set_index('cellid')

    ftc = ftc.reset_index().join(sig_cells, on=sig_cells.index.names).set_index(ftc.index.names)
    ftc['significant'] = ftc['significant'].fillna(False)

    rlf = rlf.reset_index().join(sig_cells, on=sig_cells.index.names).set_index(rlf.index.names)
    rlf['significant'] = rlf['significant'].fillna(False)

    sr = sr.reset_index().join(sig_cells, on=sig_cells.index.names).set_index(sr.index.names)
    sr['significant'] = sr['significant'].fillna(False)

    return {
        'ftc': ftc.rename(columns=renamer),
        'rlf': rlf.rename(columns=renamer),
        'sr': sr.rename(columns=renamer),
    }


def load_sr():
    return load_rates()['sr']


def load_ftc():
    return load_rates()['ftc']


def load_rlf():
    return load_rates()['rlf']
