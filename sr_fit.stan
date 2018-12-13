data {
    int<lower=1> n_cells;
    real<lower=0> sample_time_A[n_cells];
    real<lower=0> sample_time_B[n_cells];
    int<lower=0> spike_count_A[n_cells];
    int<lower=0> spike_count_B[n_cells];
}

parameters {
    real<lower=0> rate_alpha;
    real<lower=0> rate_beta;
    real<lower=0> rate_cell[n_cells];
    
    real<lower=0> rate_ratio_mean;
    real<lower=0> rate_ratio_sd;
    real<lower=0> rate_ratio_cell[n_cells];
}

transformed parameters {
    real rate_A;
    real rate_B;
    real rate_change;
    real rate_cell_A[n_cells];
    real rate_cell_B[n_cells];
    real rate_cell_change[n_cells];
    
    rate_A = rate_alpha/rate_beta;
    rate_B = rate_A * rate_ratio_mean;
    rate_change = rate_B - rate_A;
    
    rate_cell_A = rate_cell;
    for (i in 1:n_cells) {
        rate_cell_B[i] = rate_cell[i] * rate_ratio_cell[i];
        rate_cell_change[i] = rate_cell_B[i] - rate_cell_A[i];
    }
}

model {
    rate_alpha ~ gamma(0.5, 0.1);
    rate_beta ~ gamma(0.1, 0.1);
    
    rate_ratio_mean ~ normal(1, 1)T[0, ];
    rate_ratio_sd ~ normal(0, 1)T[0, ];
    
    rate_cell ~ gamma(rate_alpha, rate_beta);
    rate_ratio_cell ~ normal(rate_ratio_mean, rate_ratio_sd);
    
    for (i in 1:n_cells) {
        spike_count_A[i] ~ poisson(rate_cell[i] * sample_time_A[i]);
        spike_count_B[i] ~ poisson((rate_cell[i] * rate_ratio_cell[i]) * sample_time_B[i]);
    }
}