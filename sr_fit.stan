data {
    int<lower=1> n_cells;
    vector<lower=0>[n_cells] sample_time_A;
    vector<lower=0>[n_cells] sample_time_B;
    int<lower=0> spike_count_A[n_cells];
    int<lower=0> spike_count_B[n_cells];
}

parameters {
    real<lower=0> rate_alpha;
    real<lower=0> rate_beta;
    vector<lower=0>[n_cells] rate_cell_A;
    
    real<lower=0> rate_ratio_mean;
    real<lower=0> rate_ratio_sd;
    vector<lower=0>[n_cells] rate_ratio_cell;
}

transformed parameters {
    real rate_A;
    real rate_B;
    real rate_change;
    
    vector[n_cells] rate_cell_B;
    vector[n_cells] rate_cell_change;
    
    rate_A = rate_alpha/rate_beta;
    rate_B = rate_A * rate_ratio_mean;
    rate_change = rate_B - rate_A;
    
    rate_cell_B = rate_cell_A .* rate_ratio_cell;
    rate_cell_change = rate_cell_B - rate_cell_A;
}

model {
    vector[n_cells] lambda_A;
    vector[n_cells] lambda_B;
    
    rate_alpha ~ gamma(0.5, 0.1);
    rate_beta ~ gamma(0.1, 0.1);
    
    rate_ratio_mean ~ normal(1, 1);
    rate_ratio_sd ~ normal(0, 1);
    
    rate_cell_A ~ gamma(rate_alpha, rate_beta);
    rate_ratio_cell ~ normal(rate_ratio_mean, rate_ratio_sd);
    
    spike_count_A ~ poisson(rate_cell_A .* sample_time_A);
    spike_count_B ~ poisson(rate_cell_A .* rate_ratio_cell .* sample_time_B);
}