import numpy as np
import pickle

from support import CachedStanModel
from support import load_stan_data


if __name__ == '__main__':
    import argparse
    import socket
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude-silent', action='store_true')
    parser.add_argument('--significant-only', action='store_true')
    parser.add_argument('--band', action='store_true')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--th-bound', action='store_true')
    group.add_argument('--th-bound-delta', action='store_true')
    group.add_argument('--th-delta', action='store_true')

    parser.add_argument('--swap-pupil', action='store_true')
    args = parser.parse_args()

    fit_names = ['rl_sr']

    if args.th_bound:
        fit_names.append('th_bound')
        model = 'rl_with_sr_th_bound.stan'
    elif args.th_bound_delta:
        fit_names.append('th_bound_delta')
        model = 'rl_with_sr_th_bound_split_delta.stan'
    elif args.th_delta:
        fit_names.append('th_delta')
        model = 'rl_with_sr_split_delta.stan'
    else:
        model = 'rl_with_sr.stan'

    if args.band:
        fit_names.append('band')
        data = 'rlf_band'
    else:
        data = 'rlf'

    if args.swap_pupil:
        fit_names.append('swap_pupil')
    if args.exclude_silent:
        fit_names.append('exclude_silent')
    if args.significant_only:
        fit_names.append('significant_only')
    fit_name = '_'.join(fit_names)

    cells, data = load_stan_data(data, exclude_silent=args.exclude_silent,
                                 significant_only=args.significant_only,
                                 swap_pupil=args.swap_pupil)
    model = CachedStanModel(model)
    n_iter = 2000
    fit = model.sampling(data, iter=n_iter, control={'max_treedepth': 14},
                         sample_file=f'fits/{hostname}-{fit_name}_samples')

    with open(f'fits/{hostname}-{fit_name}-{n_iter}.pkl', 'wb') as fh:
        pickle.dump(cells, fh)
        pickle.dump(model, fh)
        pickle.dump(fit, fh)
