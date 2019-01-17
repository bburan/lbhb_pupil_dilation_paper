functions {
    real hinge(real x, real slope, real threshold, real sr) {
        real result;
        if (x < threshold) {
            result = sr;
        } else {
           result = fmax(0.01, slope * (x - threshold) + sr);
        }
        return(result);
    }
}

data {
    int<lower=1> n;
    int<lower=1> n_cells;
    
    vector[n] level;
    vector<lower=0>[n] time;
    int<lower=0> count[n];
    
    vector<lower=0>[n_cells] sr_time;
    int<lower=0> sr_count[n_cells];
    vector<lower=0>[n_cells] sr_time_pupil;
    int<lower=0> sr_count_pupil[n_cells];
    
    int data_cell_index[n_cells+1, 2];
}

parameters {
    real slope_mean;
    real<lower=0> slope_sd;
    vector[n_cells] slope_cell_mean;
    
    real<lower=0> slope_ratio_mean;
    real<lower=0> slope_ratio_sd;
    vector<lower=0>[n_cells] slope_ratio_cell;
    
    real threshold_mean;
    real<lower=0> threshold_sd;
    vector[n_cells] threshold_cell_mean;
    
    real threshold_delta_mean;
    real<lower=0> threshold_delta_sd;
    vector[n_cells] threshold_delta_cell;
    
    real<lower=0> sr_alpha;
    real<lower=0> sr_beta;
    vector<lower=0>[n_cells] sr_cell;
    
    real<lower=0> sr_ratio_mean;
    real<lower=0> sr_ratio_sd;
    vector<lower=0>[n_cells] sr_ratio_cell;
}

transformed parameters {
    vector[n_cells] threshold_cell;
    vector[n_cells] threshold_cell_pupil;
    
    vector[n_cells] sr_cell_pupil;
    vector[n_cells] sr_delta_cell;
    real sr_mean;
    real sr_mean_pupil;
    real sr_delta_mean;
    
    vector[n_cells] slope_cell;
    vector[n_cells] slope_cell_pupil;
    vector[n_cells] slope_delta_cell;
    real slope_delta_mean;
    
    slope_cell = 2 * slope_cell_mean ./ (1 + slope_ratio_cell);
    slope_cell_pupil = slope_cell .* slope_ratio_cell;
    slope_delta_cell = slope_cell_pupil - slope_cell;
    slope_delta_mean = (2 * slope_mean * (slope_ratio_mean - 1)) / (1 + slope_ratio_mean);
    
    threshold_cell = threshold_cell_mean + threshold_delta_cell/2;
    threshold_cell_pupil = threshold_cell_mean - threshold_delta_cell/2;
    
    sr_mean = sr_alpha/sr_beta;
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
    real slope;
    real threshold;
    real sr;
    
    slope_mean ~ normal(0.1, 0.1);
    slope_sd ~ normal(0, 0.1);
    slope_cell_mean ~ normal(slope_mean, slope_sd);
    
    slope_ratio_mean ~ normal(1, 0.1);
    slope_ratio_sd ~ normal(1, 0.1);
    slope_ratio_cell ~ normal(slope_ratio_mean, slope_ratio_sd);
    
    threshold_mean ~ normal(40, 5);
    threshold_sd ~ normal(0, 5);
    threshold_cell_mean ~ normal(threshold_mean, threshold_sd);
    
    threshold_delta_mean ~ normal(0, 1);
    threshold_delta_sd ~ normal(0, 1);
    threshold_delta_cell ~ normal(threshold_delta_mean, threshold_delta_sd);
    
    sr_alpha ~ gamma(0.5, 0.1);
    sr_beta ~ gamma(0.1, 0.1);
    sr_ratio_mean ~ normal(1, 1);
    sr_ratio_sd ~ normal(0, 1);
    
    sr_cell ~ gamma(sr_alpha, sr_beta);
    sr_ratio_cell ~ normal(sr_ratio_mean, sr_ratio_sd);
    
    for (c in 1:n_cells) {
        // Fit the small pupil
        lb = data_cell_index[c, 1];
        ub = data_cell_index[c, 2]-1;
        slope = slope_cell[c];
        threshold = threshold_cell[c];
        sr = sr_cell[c];
        for (i in lb:ub) {
            lambda[i] = hinge(level[i], slope, threshold, sr);
        }
        
        // Fit the large pupil
        lb = data_cell_index[c, 2];
        ub = data_cell_index[c+1, 1]-1;
        slope = slope_cell_pupil[c];
        threshold = threshold_cell_pupil[c];
        sr = sr_cell_pupil[c];
        for (i in lb:ub) {
            lambda[i] = hinge(level[i], slope, threshold, sr);
        }
    }
    
    count ~ poisson(lambda .* time);
    sr_count ~ poisson(sr_cell .* sr_time);
    sr_count_pupil ~ poisson(sr_cell_pupil .* sr_time_pupil);
}
