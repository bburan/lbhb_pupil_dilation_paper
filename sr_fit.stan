data {
    int<lower=1> n_cells;
    real<lower=0> sample_time[n_cells];
    int<lower=0> spike_count[n_cells];
    //real<lower=0> sample_time_lg[n_cells];
    //int<lower=0> spike_count_lg[n_cells];
}

parameters {
    real<lower=0> sr_alpha;
    real<lower=0> sr_beta;
    //real<lower=0> sr_alpha_delta_lg;
    //real<lower=0> sr_beta_delta_lg;
    
    real<lower=0> sr_cell[n_cells];
    //vector[n_cells] sr_cell_lg;
}

model {
    sr_alpha ~ normal(0.5, 0.1)T[0, ];
    sr_beta ~ normal(0.1, 0.1)T[0, ];
    //sr_alpha_delta_lg ~ normal(0, 0.1)T[-sr_alpha, ];
    //sr_beta_delta_lg ~ normal(0, 0.1)T[-sr_beta, ];
    
    sr_cell ~ gamma(sr_alpha, sr_beta);
    //sr_cell_lg ~ gamma(sr_alpha + sr_alpha_delta_lg, sr_beta + sr_beta_delta_lg);
    
    print(sr_alpha);
    print(sr_beta);
    print(sr_cell);
    print(sample_time);
    
    for (i in 1:n_cells) {
        spike_count[i] ~ poisson(sr_cell[i] * sample_time[i]);
        //spike_count_lg[i] ~ poisson(sr_cell_lg[i] * sample_time_lg[i]);
    }
}