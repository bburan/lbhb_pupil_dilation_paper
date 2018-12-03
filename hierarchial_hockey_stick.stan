data {
    int<lower=1> n;						// number of rows
	int<lower=1> n_cells;				// number of cells
    vector[n] x;						// level
    vector[n] y;						// spike rate
	int<lower=0, upper=1> pupil[n];		// pupil index
	int<lower=1> cell[n];				// cell index
	
	vector[n_cells] spont_small;
	vector[n_cells] spont_large;
}

parameters {
	real<lower=0> sr_alpha;
	real<lower=0> sr_beta;
	vector[n_cells] sr_cell;
	real sr_pupil;
	vector[n_cells] sr_cell_pupil;
	vector<lower=0>[n_cells] sr_cell_pupil_sd;

	real<lower=0> slope_mean;
	real slope_pupil;
	vector[n_cells] slope_cell;
	vector[n_cells] slope_cell_pupil;
	vector<lower=0>[n_cells] slope_cell_sd;
	vector<lower=0>[n_cells] slope_cell_pupil_sd;

	real threshold_mean;
	real threshold_pupil;
	vector[n_cells] threshold_cell;
	vector[n_cells] threshold_cell_pupil;
	vector<lower=0>[n_cells] threshold_cell_sd;
	vector<lower=0>[n_cells] threshold_cell_pupil_sd;

    real<lower=0> sigma;
}

	
model {
	real sr;
	real threshold;
	vector[n_cells] slope;
    vector[n] mu;
	vector[n_cells] sr_pupil_mu;
	int c;

	sr_alpha ~ normal(0.5, 0.1);
	sr_beta ~ normal(0.1, 0.1);
	sr_cell ~ gamma(sr_alpha, sr_beta);
	sr_pupil ~ normal(0, 5);
	sr_cell_pupil ~ normal(0, 5);

	slope_mean ~ normal(2, 5);
	slope_pupil ~ normal(0, 1);
	slope_cell ~ normal(0, 1);
	slope_cell_pupil ~ normal(0, 1);

	threshold_mean ~ normal(50, 5);
	threshold_pupil ~ normal(0, 5);
	threshold_cell ~ normal(0, 10);
	threshold_cell_pupil ~ normal(0, 5);

	sr_cell_pupil_sd ~ cauchy(0, 5);
	slope_cell_sd ~ normal(0, 1);
	slope_cell_pupil_sd ~ normal(0, 1);

	threshold_cell_sd ~ normal(0, 10);
	threshold_cell_pupil_sd ~ normal(0, 1);

	sigma ~ cauchy(0, 1);

	slope = slope_mean + slope_cell;

	for (i in 1:n) {
		c = cell[i];
		sr = sr_cell[c];
		#slope = slope_mean + slope_cell[c] * slope_cell_sd[c];
		threshold = threshold_mean + threshold_cell[c] * threshold_cell_sd[c];
		if (pupil[i] == 1) {
			sr = sr + sr_pupil + sr_cell_pupil[c] * sr_cell_pupil_sd[c];
			#slope = slope + slope_pupil + slope_cell_pupil[c] * slope_cell_pupil_sd[c];
			threshold = threshold + threshold_pupil + threshold_cell_pupil[c] * threshold_cell_pupil_sd[c];
		}
		if (x[i] <= threshold) {
			mu[i] = sr;
		} else {
			mu[i] = (x[i] - threshold) * slope[c];
            if (mu[i] < 0) {
                mu[i] = 0;
            }
		}
	}

	for (i in 1:n_cells) {
		sr_pupil_mu[i] = sr_cell[i] + sr_pupil + sr_cell_pupil[i] * sr_cell_pupil_sd[i];
	}

    y ~ normal(mu, sigma);
	spont_small ~ normal(sr_cell, sigma);
	spont_large ~ normal(sr_pupil_mu, sigma);
}
