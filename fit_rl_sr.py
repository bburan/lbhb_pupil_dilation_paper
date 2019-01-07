import arviz as az
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd
import pickle
import pylab as pl

from support import CachedStanModel
from support import get_metric, forest_plot, load_rates


def load_data(exclude_silent=False, significant_only=False):
    rates = load_rates()
    er = rates['rlf']
    sr = rates['sr']

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

    _, indices = np.unique(e[['cell_index', 'pupil']].values.tolist(), \
                           axis=0, return_index=True)
    indices = np.r_[indices, [len(e), -1]]
    data_cell_index = np.array(indices).reshape((-1, 2)) + 1

    return {
        'n': len(e),
        'n_cells': len(cells),
        'level': e['level'].values,
        'time': e['time'].values,
        'count': e['count'].values.astype('i'),
        'sr_count': s['count'][0].values.astype('i'),
        'sr_count_pupil': s['count'][1].values.astype('i'),
        'sr_time': s['time'][0].values,
        'sr_time_pupil': s['time'][1].values,
        'data_cell_index': data_cell_index,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude-silent', action='store_true')
    parser.add_argument('--significant-only', action='store_true')
    args = parser.parse_args()

    fit_names = ['rl_sr']
    if args.exclude_silent:
        fit_names.append('exclude_silent')
    if args.significant_only:
        fit_names.append('significant_only')
    fit_name = '_'.join(fit_names)

    data = load_data(exclude_silent=args.exclude_silent,
                     significant_only=args.significant_only)
    model = CachedStanModel('rl_with_sr.stan')
    #fit = model.sampling(data, iter=10000, control={'max_treedepth': 15})
    fit = model.sampling(data, iter=2000)

    with open(f'{fit_name}.pkl', 'wb') as fh:
        pickle.dump(model, fh)
        pickle.dump(fit, fh)

    #az.plot_trace(fit, ['sr_mean', 'sr_delta_mean', 'slope_mean',
    #                    'slope_delta_mean', 'threshold_mean',
    #                    'threshold_delta_mean', 'threshold_delta_sd'])


    #summary = az.summary(fit, credible_interval=0.9)
    #summary_95 = az.summary(fit, credible_interval=0.95)

    #f, axes = pl.subplots(1, 3, figsize=(12, 4))

    #cell_metric = get_metric(summary, 'sr_delta_cell', cells=cells)
    #pop_metric = get_metric(summary, 'sr_delta_mean')
    #forest_plot(axes[0], cell_metric, pop_metric, 'sr')

    #cell_metric = get_metric(summary, 'slope_delta_cell', cells=cells)
    #pop_metric = get_metric(summary, 'slope_delta_mean')
    #forest_plot(axes[1], cell_metric, pop_metric, 'slope')

    #cell_metric = get_metric(summary, 'threshold_delta_cell', cells=cells)
    #pop_metric = get_metric(summary, 'threshold_delta_mean')
    #forest_plot(axes[2], cell_metric, pop_metric, 'threshold')

    #f.savefig('rl_sr.eps')
