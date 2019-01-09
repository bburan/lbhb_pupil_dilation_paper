import numpy as np
import pickle

from support import CachedStanModel
from support import load_stan_data


if __name__ == '__main__':
    import argparse
    import socket
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['additive', 'multiplicative'])
    parser.add_argument('--exclude-silent', action='store_true')
    parser.add_argument('--significant-only', action='store_true')
    args = parser.parse_args()

    fit_names = [f'ftc_sr_{args.mode}']
    if args.exclude_silent:
        fit_names.append('exclude_silent')
    if args.significant_only:
        fit_names.append('significant_only')
    fit_name = '_'.join(fit_names)

    cells, data = load_stan_data('ftc', exclude_silent=args.exclude_silent,
                                 significant_only=args.significant_only)
    model = CachedStanModel(f'ftc_with_sr_{args.mode}.stan')
    fit = model.sampling(data, iter=2000, control={'max_treedepth': 12},
                         sample_file=f'fits/{hostname}-{fit_name}_samples')

    with open(f'fits/{hostname}-{fit_name}.pkl', 'wb') as fh:
        pickle.dump(cells, fh)
        pickle.dump(model, fh)
        pickle.dump(fit, fh)