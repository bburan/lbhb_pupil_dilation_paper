data {
    int<lower=1> n;
    int<lower=1> n_cells;
    int<lower=1> cell[n];
    vector[n] freq;
    int<lower=0> spike_count[n];
    vector[n] sample_time;
    int<lower=0> spont_count[n_cells];
    real<lower=0> spont_time[n_cells];
}

parameters {
    real bf_mean;
    real<lower=0> bf_sd;
    real<lower=0> bf[n_cells];
    
    real<lower=0> bandwidth_alpha;
    real<lower=0> bandwidth_beta;
    real<lower=0> bandwidth[n_cells];
    
    real gain_mean;
    real<lower=0> gain_sd;
    real gain[n_cells];
    
    real<lower=0> offset_alpha;
    real<lower=0> offset_beta;
    real<lower=0> offset[n_cells];
}

model {
    vector[n] lambda;
    vector[n_cells] lambda_sr;
    int c;
    
    bf_mean ~ normal(7.25, 1.0);
    bf_sd ~ normal(0, 1.0)T[0, ];
    
    bandwidth_alpha ~ normal(2, 0.5)T[0, ];
    bandwidth_beta ~ normal(4, 0.5)T[0, ];
    
    gain_mean ~ gamma(2.5, 0.125);
    gain_sd ~ normal(0, 20)T[0, ];
    gain ~ normal(gain_mean, gain_sd);
    
    offset_alpha ~ normal(0.5, 0.1)T[0, ];
    offset_beta ~ normal(0.1, 0.1)T[0, ];
    offset ~ gamma(offset_alpha, offset_beta);
    
    for (i in 1:n_cells) {
        bandwidth[i] ~ gamma(bandwidth_alpha, bandwidth_beta);
        bf[i] ~ normal(bf_mean, bf_sd)T[3, 11];
        gain[i] ~ normal(gain_mean, gain_sd)T[-offset[i], ];
    }
    
    for (i in 1:n) {
        c = cell[i];
        lambda[i] = exp(-0.5*square((freq[i]-bf[c])/bandwidth[c]));
        lambda[i] = offset[c] + gain[c] * lambda[i];
        lambda[i] = lambda[i] * sample_time[i];
    }
    for (i in 1:n_cells) {
        lambda_sr[i] = offset[i] * spont_time[i];
    }
    print(gain_mean)
    print(gain)
    spike_count ~ poisson(lambda);
    spont_count ~ poisson(lambda_sr);
}