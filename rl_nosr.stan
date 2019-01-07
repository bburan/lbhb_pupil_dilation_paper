data {
    int<lower=1> n;
    int<lower=1> n_cells;
    int<lower=1> cell_index[n];
    
    vector[n] level;
    int<lower=0, upper=1> pupil[n];
    real<lower=0> time[n];
    int<lower=0> count[n];
    
    real<lower=0> sr[n_cells];
    real<lower=0> sr_pupil[n_cells];
}

parameters {
    real slope_mean;
    real<lower=0> slope_sd;
    real slope_cell[n_cells];
    
    real slope_delta_mean;
    real<lower=0> slope_delta_sd;
    real slope_delta_cell[n_cells];
    
    real threshold_mean;
    real<lower=0> threshold_sd;
    real threshold_cell[n_cells];
    
    real threshold_delta_mean;
    real<lower=0> threshold_delta_sd;
    real threshold_delta_cell[n_cells];
}

model {
    real lambda[n];
    
    int c;
    real l;
    real sr_n;
    real th_n;
    real s_n;
    
    slope_mean ~ normal(0.1, 0.1);
    slope_sd ~ normal(0, 0.1)T[0, ];
    slope_cell ~ normal(slope_mean, slope_sd);
    
    slope_delta_mean ~ normal(0, 0.1);
    slope_delta_sd ~ normal(0, 0.1)T[0, ];
    slope_delta_cell ~ normal(slope_delta_mean, slope_delta_sd);
    
    threshold_mean ~ normal(40, 20);
    threshold_sd ~ normal(0, 20)T[0, ];
    threshold_cell ~ normal(threshold_mean, threshold_sd);
    
    threshold_delta_mean ~ normal(0, 5);
    threshold_delta_sd ~ normal(0, 5)T[0, ];
    threshold_delta_cell ~ normal(threshold_delta_mean, threshold_delta_sd);
    
    for (i in 1:n) {
        c = cell_index[i];
        if (pupil[i] == 0) { 
            sr_n = sr[c]; 
            th_n = threshold_cell[c];
            s_n = slope_cell[c];
        } else { 
            sr_n = sr_pupil[c]; 
            th_n = threshold_cell[c] + threshold_delta_cell[c];
            s_n = slope_cell[c] + slope_delta_cell[c];
        }

        if (level[i] <= th_n) {
            lambda[i] = sr_n * time[i]; 
        } else {
            l = (s_n * (level[i]-th_n) + sr_n) * time[i];
            lambda[i] = fmax(l, 0.01);
        }
    }
    
    count ~ poisson(lambda);
}