from hashlib import md5
import pickle

import numpy as np
import pandas as pd
import pystan
import seaborn as sns


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
    
    significant = rlf.reset_index().groupby(['cellid'])['significant'].first()
    
    ftc = ftc.rename(columns=renamer)
    rlf = rlf.rename(columns=renamer)
    sr = sr.rename(columns=renamer)

    m = sr['time'] != 0
    sr = sr.loc[m]
    
    m = rlf['time'] != 0
    rlf = rlf.loc[m]
    
    m = ftc['time'] != 0
    ftc = ftc.loc[m]

    return {
        'ftc': ftc,
        'rlf': rlf,
        'sr': sr,
        'significant': significant,
    }


def load_sr():
    return load_rates()['sr']


def load_ftc():
    return load_rates()['ftc']


def load_rlf():
    return load_rates()['rlf']


def get_metric(summary, metric, index=None, cells=None):
    x = summary[metric].to_series()
    if x.index.nlevels == 2:
        x = x.unstack('metric')
    if cells is not None:
        index = pd.Index(cells, name='cellid')
    if index is not None:
        x.index = index
    return x


def get_color(row, lb_label, ub_label):
    if row['gelman-rubin statistic'] > 1.1:
        return 'red'
    if (row[lb_label] > 0) or (row[ub_label] < 0):
        return 'green'
    return 'gray'


def forest_plot(ax, cell_metric, pop_metric, measure):
    cell_metric = cell_metric.sort_values('mean')
    ci_label = ['hpd 5.00%', 'hpd 95.00%']
    
    color = get_color(pop_metric, *ci_label)
    ax.axvspan(*pop_metric[ci_label], facecolor=color, alpha=0.5)
    ax.axvline(pop_metric['mean'], color=color)
    n_sig = 0
    for i, (cell_index, row) in enumerate(cell_metric.iterrows()):
        lw = 0.5
        color = get_color(row, *ci_label)
        if color == 'green':
            n_sig += 1
        ax.plot(row[ci_label], [i, i], '-', color=color, lw=lw)
        ax.plot(row[['mean']], [i], 'o', color=color)
        
    title = f'Change in {measure} (lg. re sm. pupil)'
    n_sig = f'{n_sig} sig. out of {len(cell_metric)}'
    pop_stat = f'Mean change {pop_metric["mean"]:.2f} (90% CI {pop_metric[ci_label[0]]:.2f} to {pop_metric[ci_label[1]]:.2f})'
    ax.set_xlabel(f'{title}\n{n_sig}\n{pop_stat}')
    sns.despine(ax=ax, top=True, left=True, right=True, bottom=False)
    ax.yaxis.set_ticks_position('none')
    ax.yaxis.set_ticks([])
    ax.grid()
    return ax

