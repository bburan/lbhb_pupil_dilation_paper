data {
    int<lower=1> n;
    vector[n] evoked_level;
	int<lower=0, upper=1> pupil[n];
	real<lower=0> evoked_time[n];
    int<lower=0> evoked_count[n];
	int<lower=0> sr_count[2];
	real<lower=0> sr_time[2];
}


parameters {
    real slope;
	real slope_pupil_delta;
    real<lower=0> sr;
	real sr_pupil_delta;
    real threshold;
	real threshold_pupil_delta;
}


model {
    real slope_pupil;
    real sr_pupil;
    real threshold_pupil;
    real l;
	real lambda[n];
	real sr_lambda[2];
	real i_sr;
	real i_threshold;
	real i_slope;
    
    slope_pupil = slope + slope_pupil_delta;
    sr_pupil = sr + sr_pupil_delta;
    threshold_pupil = threshold + threshold_pupil_delta;

    sr ~ gamma(0.5, 0.1);
    sr_pupil ~ gamma(0.5, 0.1);
	sr_pupil_delta ~ normal(0, 10);
    
    slope ~ normal(0.1, 0.25);
	slope_pupil_delta ~ normal(0, 0.25);
    
	threshold ~ normal(40, 40);
	threshold_pupil_delta ~ normal(0, 20);
    
	for (i in 1:n) {
		i_sr = sr + sr_pupil_delta * pupil[i];
		i_threshold = threshold + threshold_pupil_delta * pupil[i];
		i_slope = slope + slope_pupil_delta * pupil[i];

		if (evoked_level[i] <= i_threshold) {
            l = i_sr * evoked_time[i]; 
            lambda[i] = l;
		} else {
            l = (i_slope*(evoked_level[i]-i_threshold) + i_sr) * evoked_time[i];
            lambda[i] = fmax(l, 0.01);
		}
	}
    
	sr_lambda[1] = sr * sr_time[1];
    l = (sr + sr_pupil_delta) * sr_time[2];
	sr_lambda[2] = fmax(l, 0.01);
    evoked_count ~ poisson(lambda);
	sr_count ~ poisson(sr_lambda);
}