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
    
    vector<lower=0>[n_cells] sr_cell;
    vector<lower=0>[n_cells] sr_cell_pupil;
    
    int data_cell_index[n_cells+1, 2];
}

parameters {
    real slope_mean;
    real<lower=0> slope_sd;
    vector[n_cells] slope_cell;
    
    real slope_delta_mean;
    real<lower=0> slope_delta_sd;
    vector[n_cells] slope_delta_cell;
    
    real threshold_mean;
    real<lower=0> threshold_sd;
    vector[n_cells] threshold_cell;
    
    real threshold_delta_mean;
    real<lower=0> threshold_delta_sd;
    vector[n_cells] threshold_delta_cell;
}

transformed parameters {
    vector[n_cells] slope_cell_pupil;
    vector[n_cells] threshold_cell_pupil;
    
    slope_cell_pupil = slope_cell + slope_delta_cell;
    threshold_cell_pupil = threshold_cell + threshold_delta_cell;
}

model {
    vector[n] lambda;
    int lb;
    int ub;
    real slope;
    real threshold;
    real sr;
    
    slope_mean ~ normal(0.1, 0.1);
    slope_sd ~ normal(0, 0.1);
    slope_cell ~ normal(slope_mean, slope_sd);
    
    slope_delta_mean ~ normal(0, 0.1);
    slope_delta_sd ~ normal(0, 0.1);
    slope_delta_cell ~ normal(slope_delta_mean, slope_delta_sd);
    
    threshold_mean ~ normal(40, 20);
    threshold_sd ~ normal(0, 20);
    threshold_cell ~ normal(threshold_mean, threshold_sd);
    
    threshold_delta_mean ~ normal(0, 5);
    threshold_delta_sd ~ normal(0, 5);
    threshold_delta_cell ~ normal(threshold_delta_mean, threshold_delta_sd);
    
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
}