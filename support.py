from hashlib import md5
import pickle
import socket

import numpy as np
import pandas as pd
import pystan
import seaborn as sns


def CachedStanModel(model_file, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    hostname = socket.gethostname()
    with open(model_file, 'rb') as fh:
        model_code = fh.read().decode('utf8')
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = f'cached_models/{hostname}-cached-model-{code_hash}.pkl'
    else:
        cache_fn = f'cached_models/{hostname}-cached-model-{model_name}-{code_hash}.pkl'
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


def _load_spreadsheet(filename, swap_pupil=False):
    x = pd.read_csv(filename)
    x.columns = [c.replace(' ', '') for c in x.columns]
    cols = set(c.rsplit('_', 1)[0] for c in x.columns if '_' in c)
    x = pd.wide_to_long(x, cols, 'cellid', 'idx', sep='_').dropna()
    x['pupil'] -= 1
    if swap_pupil:
        x['pupil'] = 1 - x['pupil']
    return x
    
    
def _reformat(x, base, sig_cells):
    metric = 'frequency' if base == 'ftc' else 'level'
    x = x.reset_index().set_index(['cellid', 'pupil', metric])[[f'{base}_count', f'{base}_time']].sort_index()
    m = x[f'{base}_time'] > 0
    x = x.loc[m]
    x = x.reset_index().join(sig_cells, on=sig_cells.index.names).set_index(x.index.names)
    x['significant'] = x['significant'].fillna(False)
    x = x.rename(columns=renamer)
    return x

    
def load_rates(swap_pupil=False):
    # NOTE: Pupil should be 0 (small) or 1 (large)
    # Frequency should be octave-spaced

    sig_cells = pd.read_csv('psth_sig_cellids.csv')
    sig_cells['significant'] = True
    sig_cells = sig_cells.set_index('cellid')

    # Load FTC
    ftc = _load_spreadsheet('frequency_tuning_curves_for_bburan.csv', swap_pupil)
    ftc['frequency'] = np.log2(ftc['frequency'])
    sr = ftc.groupby(['cellid', 'pupil'])[['spont_count', 'spont_time']].first().sort_index()
    ftc = _reformat(ftc, 'ftc', sig_cells)
    
    ftc_band = _load_spreadsheet('frequency_tuning_curves_for_bburan_level_band.csv', swap_pupil)
    ftc_band['frequency'] = np.log2(ftc_band['frequency'])
    ftc_band = _reformat(ftc_band, 'ftc', sig_cells)
    
    rlf = _load_spreadsheet('rate_level_functions_for_bburan.csv', swap_pupil)
    rlf = _reformat(rlf, 'rlf', sig_cells)

    rlf_band = _load_spreadsheet('rate_level_functions_for_bburan_freq_band.csv', swap_pupil)
    rlf_band = _reformat(rlf_band, 'rlf', sig_cells)
    
    sr = sr.rename(columns=renamer)
    m = sr['time'] != 0
    sr = sr.loc[m]

    return {
        'ftc': ftc,
        'ftc_band': ftc_band,
        'rlf': rlf,
        'rlf_band': rlf_band,
        'sr': sr,
    }


def load_stan_data(which='rlf', exclude_silent=False, significant_only=False,
                   o=None, n=None, swap_pupil=False):
    rates = load_rates(swap_pupil)
    sr = rates['sr']

    if which == 'rlf':
        er = rates['rlf']
        key = 'level'
    elif which == 'rlf_band':
        er = rates['rlf_band']
        key = 'level'
    elif which == 'ftc':
        er = rates['ftc']
        key = 'frequency'
    elif which == 'ftc_band':
        er = rates['ftc_band']
        key = 'frequency'

    if o is not None and n is not None:
        cells = er.reset_index()['cellid'].unique()
        lb = cells[o]
        ub = cells[o+n-1]
        sr = sr.loc[lb:ub]
        er = er.loc[lb:ub]

    if exclude_silent:
        spike_counts = er['count'].groupby(['cellid', 'pupil']).sum()
        m = spike_counts == 0
        exclude = spike_counts.loc[m].unstack().index.values.tolist()
        sr = sr.drop(exclude)
        er = er.drop(exclude)

    if significant_only:
        er = er.query('significant')
        sr = sr.query('significant')

    e = er.reset_index()
    s = sr.reset_index()

    cells = e['cellid'].unique()
    cell_map = {c: i+1 for i, c in enumerate(cells)}
    e['cell_index'] = e['cellid'].apply(cell_map.get).values
    s['cell_index'] = s['cellid'].map(cell_map.get)
    s = s.set_index(['cell_index', 'pupil']) \
        .sort_index()[['count', 'time']].unstack()

    z, indices = np.unique(e[['cell_index', 'pupil']].values.tolist(), \
                           axis=0, return_index=True)

    indices = np.r_[indices, [len(e), -1]]
    data_cell_index = np.array(indices).reshape((-1, 2)) + 1

    return cells, {
        'n': len(e),
        'n_cells': len(cells),
        key: e[key].values,
        'time': e['time'].values,
        'count': e['count'].values.astype('i'),
        'sr_count': s['count'][0].values.astype('i'),
        'sr_count_pupil': s['count'][1].values.astype('i'),
        'sr_time': s['time'][0].values,
        'sr_time_pupil': s['time'][1].values,
        'data_cell_index': data_cell_index,
    }


def get_metric(summary, metric, index=None, cells=None, sig_ref=0):
    x = summary[metric].to_series()
    if x.index.nlevels == 2:
        x = x.unstack('metric')
        x['change'] = '='
        x.loc[x['hpd 5%'] > sig_ref, 'change'] = '+'
        x.loc[x['hpd 95%'] < sig_ref, 'change'] = '-'
    else:
        x['change'] = '='
        if x['hpd 5%'] > sig_ref:
            x['change'] = '+'
        elif x['hpd 95%'] < sig_ref:
            x['change'] = '-'
    if cells is not None:
        index = pd.Index(cells, name='cellid')
    if index is not None:
        x.index = index

    return x


def get_color(row, lb_label, ub_label, ref=0):
    if row['r_hat'] > 1.1:
        return 'red'
    if ref is None:
        return 'gray'
    if (row[lb_label] > ref) or (row[ub_label] < ref):
        return 'green'
    return 'gray'


def forest_plot(ax, cell_metric, pop_metric, title, ci=90, ref=0):
    cell_metric = cell_metric.sort_values('mean')
    if ci == 90:
        ci_label = ['hpd 5%', 'hpd 95%']
    elif ci == 95:
        ci_label = ['hpd 2.50%', 'hpd 97.50%']

    color = get_color(pop_metric, *ci_label, ref=ref)
    ax.axvspan(*pop_metric[ci_label], facecolor=color, alpha=0.5)
    ax.axvline(pop_metric['mean'], color=color)
    n_sig = 0
    for i, (cell_index, row) in enumerate(cell_metric.iterrows()):
        lw = 0.5
        color = get_color(row, *ci_label, ref=ref)
        if color == 'green':
            n_sig += 1
        ax.plot(row[ci_label], [i, i], '-', color=color, lw=lw)
        ax.plot(row[['mean']], [i], 'o', color=color)

    n_sig = f'{n_sig} sig. out of {len(cell_metric)}'
    pop_stat = f'Mean {pop_metric["mean"]:.2f} ({ci}% CI {pop_metric[ci_label[0]]:.2f} to {pop_metric[ci_label[1]]:.2f})'
    ax.set_xlabel(f'{title}\n{n_sig}\n{pop_stat}')
    sns.despine(ax=ax, top=True, left=True, right=True, bottom=False)
    ax.yaxis.set_ticks_position('none')
    ax.yaxis.set_ticks([])
    ax.grid()
    return ax
