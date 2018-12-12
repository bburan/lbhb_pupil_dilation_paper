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
    real<lower=0> bf[n_cells];
    real<lower=0> bw[n_cells];
    real gain[n_cells];
    real<lower=0> offset[n_cells];
}

model {
    vector[n] lambda;
    vector[n_cells] lambda_sr;
    int c;
    
    for (i in 1:n_cells) {
        bf[i] ~ normal(7.25, 2.0);
        bw[i] ~ normal(1, 1)T[0, ];
        offset[i] ~ gamma(0.65, 0.07);
        gain[i] ~ normal(35, 35)T[-offset[i], ];
    }
    
    for (i in 1:n) {
        c = cell[i];
        lambda[i] = exp(-0.5*square((freq[i]-bf[c])/bw[c]));
        lambda[i] = offset[c] + gain[c] * lambda[i];
        lambda[i] = lambda[i] * sample_time[i];
    }
    for (i in 1:n_cells) {
        lambda_sr[i] = offset[i] * spont_time[i];
    }
    spike_count ~ poisson(lambda);
    spont_count ~ poisson(lambda_sr);
}