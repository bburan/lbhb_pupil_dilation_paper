data {
    int<lower=1> n;						// number of rows
	int<lower=1> n_cells;				// number of cells
    vector[n] x;						// level
    vector[n] y;						// spike rate
	int<lower=0, upper=1> pupil[n];		// pupil index
	int<lower=1> cell[n];				// cell index
	vector[n_cells] spont;				// SR per cell
}

parameters {
	real<lower=0> sr_mean;
	real sr_pupil;
	vector[n_cells] sr_cell;
	vector[n_cells] sr_cell_pupil;
	vector<lower=0>[n_cells] sr_cell_sd;
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
    real<lower=0> sigma_sr;
}

	
model {
	real sr;
	real threshold;
	real slope;
    vector[n] mu;
	int c;

	sr_mean ~ normal(10, 10)T[0, ];
	sr_pupil ~ normal(0, 5);
	sr_cell ~ normal(0, 5);
	sr_cell_pupil ~ normal(0, 5);

	slope_mean ~ normal(2, 5)T[0, ];
	slope_pupil ~ normal(0, 1);
	slope_cell ~ normal(0, 1);
	slope_cell_pupil ~ normal(0, 1);

	threshold_mean ~ normal(50, 5);
	threshold_pupil ~ normal(0, 5);
	threshold_cell ~ normal(0, 10);
	threshold_cell_pupil ~ normal(0, 5);

	// T is not vectorized, so we need to use a for loop.
	for (i in 1:n_cells) {
		sr_cell_sd[i] ~ normal(0, 5)T[0, ];
		sr_cell_pupil_sd[i] ~ normal(0, 5)T[0, ];

		slope_cell_sd[i] ~ normal(0, 1)T[0, ];
		slope_cell_pupil_sd[i] ~ normal(0, 1)T[0, ];

		threshold_cell_sd[i] ~ normal(0, 10)T[0, ];
		threshold_cell_pupil_sd[i] ~ normal(0, 1)T[0, ];
	}

	sigma ~ normal(0, 1)T[0, ];
	sigma_sr ~ normal(0, 1)T[0, ];

	for (i in 1:n) {
		c = cell[i];
		sr = sr_mean + sr_cell[c] * sr_cell_sd[c];
		slope = slope_mean + slope_cell[c] * slope_cell_sd[c];
		threshold = threshold_mean + threshold_cell[c] * threshold_cell_sd[c];
		if (pupil[i] == 1) {
			sr = sr + sr_pupil + sr_cell_pupil[c] * sr_cell_pupil_sd[c];
			slope = slope + slope_pupil + slope_cell_pupil[c] * slope_cell_pupil_sd[c];
			threshold = threshold + threshold_pupil + threshold_cell_pupil[c] * threshold_cell_pupil_sd[c];
		}
		mu[i] = fmax(sr, (x[i] - threshold) * slope);
	}

    y ~ normal(mu, sigma);
	spont ~ normal(sr_mean + sr_cell[c] * sr_cell_sd[c], sigma_sr);
}
