data {
    int<lower=1> n;
    int<lower=1> n_cells;
    int<lower=1> cell[n];
    vector[n] freq;
    vector[n_cells] bf_lb;
    vector[n_cells] bf_ub;
    int<lower=0> spike_count[n];
    vector[n] sample_time;
    int<lower=0> spont_count[n_cells];
    real<lower=0> spont_time[n_cells];
}

parameters {
    real<lower=0> bf[n_cells];
    real<lower=0> bandwidth[n_cells];
    real gain[n_cells];
    real<lower=0> offset[n_cells];
}

model {
    vector[n] lambda;
    vector[n_cells] lambda_sr;
    int c;
    
    for (i in 1:n_cells) {
        bf[i] ~ normal(7.25, 2.0)T[bf_lb[i], bf_ub[i]];
        bandwidth[i] ~ normal(1, 1)T[0, ];
        offset[i] ~ normal(5, 5)T[0, ];
        gain[i] ~ normal(50, 20)T[-offset[i], ];
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
    print(offset)
    print(gain)
    print(lambda)
    spike_count ~ poisson(lambda);
    spont_count ~ poisson(lambda_sr);
}