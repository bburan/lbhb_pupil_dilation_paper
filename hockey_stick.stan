data {
    int<lower=1> n;
    int<lower=1> n_cells;
    int<lower=1> cell_index[n];
    
    vector[n] evoked_level;
    int<lower=0, upper=1> pupil[n];
    real<lower=0> evoked_time[n];
    int<lower=0> evoked_count[n];
    
    int<lower=0> sr_count[2*n_cells];
    real<lower=0> sr_time[2*n_cells];
}


parameters {
    real slope_mean;
    real<lower=0> slope_sd;
    real slope_pupil_delta_mean;
    real<lower=0> slope_pupil_delta_sd;
    real slope[n_cells];
    real slope_pupil_delta[n_cells];
    
    real<lower=0> sr_alpha;
    real<lower=0> sr_beta;
    real<lower=0> sr_pupil_alpha;
    real<lower=0> sr_pupil_beta;
    real<lower=0> sr[n_cells];
    real<lower=0> sr_pupil[n_cells];
    
    real threshold_mean;
    real<lower=0> threshold_sd;
    real threshold_pupil_delta_mean;
    real<lower=0> threshold_pupil_delta_sd;
    real threshold[n_cells];
    real threshold_pupil_delta[n_cells];
}


model {
    real lambda[n];
    real sr_lambda[2*n_cells];
    
    int c;
    real l;
    real i_sr;
    real i_threshold;
    real i_slope;
    
    sr_alpha ~ normal(0.5, 0.1)T[0, ];
    sr_beta ~ normal(0.1, 0.1)T[0, ];
    sr ~ gamma(sr_alpha, sr_beta);
    
    sr_pupil_alpha ~ normal(0.5, 0.1)T[0, ];
    sr_pupil_beta ~ normal(0.1, 0.1)T[0, ];
    sr_pupil ~ gamma(sr_pupil_alpha, sr_pupil_beta);
    
    slope_mean ~ normal(0.1, 0.1);
    slope_sd ~ normal(0, 0.1)T[0, ];
    slope ~ normal(slope_mean, slope_sd);
    
    slope_pupil_delta_mean ~ normal(0, 0.1);
    slope_pupil_delta_sd ~ normal(0, 0.1)T[0, ];
    slope_pupil_delta ~ normal(slope_pupil_delta_mean, slope_pupil_delta_sd);
    
    threshold_mean ~ normal(40, 20);
    threshold_sd ~ normal(0, 20)T[0, ];
    threshold ~ normal(threshold_mean, threshold_sd);
    
    threshold_pupil_delta_mean ~ normal(0, 5);
    threshold_pupil_delta_sd ~ normal(0, 5)T[0, ];
    threshold_pupil_delta ~ normal(threshold_pupil_delta_mean, threshold_pupil_delta_sd);
    
    for (i in 1:n) {
        c = cell_index[i];
        if (pupil[i] == 1) {
            i_sr = sr_pupil[c];
        } else {
            i_sr = sr[c];
        }
        
        i_threshold = threshold[c] + threshold_pupil_delta[c] * pupil[i];
        i_slope = slope[c] + slope_pupil_delta[c] * pupil[i];

        if (evoked_level[i] <= i_threshold) {
            lambda[i] = i_sr * evoked_time[i]; 
        } else {
            l = (i_slope*(evoked_level[i]-i_threshold) + i_sr) * evoked_time[i];
            lambda[i] = fmax(l, 0.01);
        }
    }
    
    for (i in 1:n_cells) {
        sr_lambda[i] = sr[i] * sr_time[i];
        sr_lambda[i + n_cells] = sr_pupil[i] * sr_time[i + n_cells];
    }
    evoked_count ~ poisson(lambda);
    sr_count ~ poisson(sr_lambda);
}