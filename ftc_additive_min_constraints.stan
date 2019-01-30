functions {
    real gauss(real x, real gain, real bf, real bw, real sr) {
        return(fmax(0.01, sr + gain * exp(-0.5 * square((x - bf)/bw))));
    }
}

data {
    int<lower=1> n;
    int<lower=1> n_cells;
    
    vector[n] frequency;
    vector<lower=0>[n] time;
    int<lower=0> count[n];

    vector<lower=0>[n_cells] sr_time;
    int<lower=0> sr_count[n_cells];
    vector<lower=0>[n_cells] sr_time_pupil;
    int<lower=0> sr_count_pupil[n_cells];
    
    int data_cell_index[n_cells+1, 2];
}

parameters {
    vector[n_cells] bf_cell;
    vector[n_cells] gain_cell;
    vector<lower=0>[n_cells] bw_cell;
    
    real bf_delta_mean;
    real<lower=0> bf_delta_sd;
    vector[n_cells] bf_delta_cell;
    
    real<lower=0> gain_ratio_mean;
    real<lower=0> gain_ratio_sd;
    vector<lower=0>[n_cells] gain_ratio_cell;
    
    real<lower=0> bw_ratio_mean;
    real<lower=0> bw_ratio_sd;
    vector<lower=0>[n_cells] bw_ratio_cell;
    
    real<lower=0> sr_alpha;
    real<lower=0> sr_beta;
    vector<lower=0>[n_cells] sr_cell;
    
    real<lower=0> sr_ratio_mean;
    real<lower=0> sr_ratio_sd;
    vector<lower=0>[n_cells] sr_ratio_cell;
}

transformed parameters {
    vector[n_cells] bf_cell_pupil;
    vector[n_cells] bw_cell_pupil;
    vector[n_cells] bw_delta_pupil;
    vector[n_cells] bw_delta_cell;
    vector[n_cells] gain_cell_pupil;
    vector[n_cells] gain_delta_cell;
    vector[n_cells] sr_cell_pupil;
    vector[n_cells] sr_delta_cell;
    
    real sr_mean;
    real sr_mean_pupil;
    real sr_delta_mean;
    
    bf_cell_pupil = bf_cell + bf_delta_cell;
    bw_cell_pupil = bw_cell .* bw_ratio_cell;
    bw_delta_pupil = bw_cell_pupil - bw_cell;
    bw_delta_cell = bw_cell_pupil - bw_cell;
    gain_cell_pupil = gain_cell .* gain_ratio_cell;
    gain_delta_cell = gain_cell_pupil - gain_cell;
    
    sr_mean = sr_alpha / sr_beta;
    sr_mean_pupil = sr_mean * sr_ratio_mean;
    sr_delta_mean = sr_mean_pupil - sr_mean;
    
    sr_cell_pupil = sr_cell .* sr_ratio_cell;
    sr_delta_cell = sr_cell_pupil - sr_cell;
    
}

model {
    vector[n] lambda;
    vector[n_cells] lambda_sr;
    vector[n_cells] lambda_sr_pupil;
    
    int lb;
    int ub;
    real bf;
    real bw;
    real gain;
    real sr;
    
    sr_alpha ~ gamma(0.7, 0.1);
    sr_beta ~ gamma(0.1, 0.01);
    sr_ratio_mean ~ normal(1.2, 0.06);
    sr_ratio_sd ~ normal(0.65, 0.06);
    
    sr_cell ~ gamma(sr_alpha, sr_beta);
    sr_ratio_cell ~ normal(sr_ratio_mean, sr_ratio_sd);
    
    bf_cell ~ normal(10, 5);
    bw_cell ~ normal(0.5, 0.1);
    gain_cell ~ normal(15, 15);
    
    bf_delta_mean ~ normal(0, 0.1);
    bf_delta_sd ~ normal(0, 0.1);
    bf_delta_cell ~ normal(bf_delta_mean, bf_delta_sd);
    
    
    bw_ratio_mean ~ normal(0, 0.1);
    bw_ratio_sd ~ normal(0, 0.1);
    bw_ratio_cell ~ normal(bw_ratio_mean, bw_ratio_sd);
    
    gain_ratio_mean ~ normal(1, 0.1);
    gain_ratio_sd ~ normal(0, 0.1);
    gain_ratio_cell ~ normal(gain_ratio_mean, gain_ratio_sd);
    
    for (c in 1:n_cells) {
        // Fit the small pupil
        lb = data_cell_index[c, 1];
        ub = data_cell_index[c, 2]-1;
        bf = bf_cell[c];
        bw = bw_cell[c];
        gain = gain_cell[c];
        sr = sr_cell[c];
        for (i in lb:ub) {
            lambda[i] = gauss(frequency[i], gain, bf, bw, sr);
        }
        
        // Fit the large pupil
        lb = data_cell_index[c, 2];
        ub = data_cell_index[c+1, 1]-1;
        bf = bf_cell_pupil[c];
        gain = gain_cell_pupil[c];
        sr = sr_cell_pupil[c];
        bw = bw_cell_pupil[c];
        for (i in lb:ub) {
            lambda[i] = gauss(frequency[i], gain, bf, bw, sr);
        }
    }
    
    count ~ poisson(lambda .* time);
    sr_count ~ poisson(sr_cell .* sr_time);
    sr_count_pupil ~ poisson(sr_cell_pupil .* sr_time_pupil);
}
