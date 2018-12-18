data {
    int<lower=1> n;
    int<lower=1> n_cells;
    int<lower=1> cell_index[n];
    int<lower=0, upper=1> pupil[n];
    
    real freq[n];
    real<lower=0> time[n];
    int<lower=0> count[n];
    vector[n_cells] sr;
    vector[n_cells] sr_pupil;
}

parameters {
    real bf_cell[n_cells];
    real gain_cell[n_cells];
    real bw_cell[n_cells];
    
    real bf_delta_mean;
    real<lower=0> bf_delta_sd;
    real bf_cell_delta[n_cells];
    
    real gain_delta_mean;
    real<lower=0> gain_delta_sd;
    real gain_cell_delta[n_cells];
    
    real bw_delta_mean;
    real<lower=0> bw_delta_sd;
    real bw_cell_delta[n_cells];
}

transformed parameters {
    real gain_cell[n_cells];
    -sr * 

}

model {
    vector[n] lambda;
    int c;
    
    real gauss;
    real bf_i;
    real bw_i;
    real gain_i;
    real sr_i;
    
    bf_cell ~ normal(10, 5);
    bw_cell ~ normal(0, 1);
    gain_cell ~ normal(1, 1);
    
    bf_delta_mean ~ normal(0, 1);
    bf_delta_sd ~ normal(0, 1);
    bf_cell_delta ~ normal(bf_delta_mean, bf_delta_sd);
    
    bw_delta_mean ~ normal(0, 1);
    bw_delta_sd ~ normal(0, 1);
    bw_cell_delta ~ normal(bw_delta_mean, bw_delta_sd);
    
    //gain_delta_mean ~ normal(0, 1);
    //gain_delta_sd ~ normal(0, 1);
    //gain_cell_delta ~ normal(gain_delta_mean, gain_delta_sd);
    
    gain_ratio_mean ~ normal(1, 1);
    gain_ratio_sd ~ normal(0, 1);
    gain_cell_ratio ~ normal(gain_ratio_mean, gain_ratio_sd);
    
    for (i in 1:n) {
        c = cell_index[i];
        if (pupil[i] == 0) {
            sr_i = sr[c];
            bf_i = bf_cell[c];
            bw_i = exp(bw_cell[c]);
            gain_i = -sr_i + exp(gain_cell[c]);
        } else {
            sr_i = sr_pupil[c];
            bf_i = bf_cell[c] + bf_cell_delta[c];
            bw_i = exp(bw_cell[c] + bw_cell_delta[c]);
            gain_i = -sr_i + exp(gain_cell[c]) * np.exp(gain_cell_ratio[c]);
        }
        gauss = gain_i * exp(-0.5*square((freq[i]-bf_i)/bw_i));
        lambda[i] = (sr_i + gauss) * time[i];
    }
    count ~ poisson(lambda);
}