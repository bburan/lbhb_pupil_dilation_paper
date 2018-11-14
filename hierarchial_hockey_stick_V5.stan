data {
    int<lower=1> n;						# number of rows
	int<lower=1> n_cells;				# number of cells
    vector[n] x;						# level
    vector[n] y;						# spike rate
	int<lower=1, upper=2> pupil[n];		# pupil index
	int<lower=1> cell[n];				# cell index
}

parameters {
	real<lower=0> sr_alpha;
	real<lower=0> sr_beta;
	real<lower=0> s_alpha;
	real<lower=0> s_beta;
	real i_mean;
	real<lower=0> i_sd;

	real<lower=0> sr[n_cells];
	real<lower=0> slope[n_cells];
	real intercept[n_cells];

	vector[n_cells] sr_pupil[2];
	vector[n_cells] slope_pupil[2];
	vector[n_cells] intercept_pupil[2];
	
    real sigma;
}

transformed parameters {
	real<lower=0> sr_cell_pupil[2, n_cells];
	real<lower=0> s_cell_pupil[2, n_cells];
	real i_cell_pupil[2, n_cells];

	for (c in 1:n_cells) {
		for (p in 1:2) {
			sr_cell_pupil[p, c] = sr[c] + sr_pupil[p, c];
			s_cell_pupil[p, c] = slope[c] + slope_pupil[p, c];
			i_cell_pupil[p, c] = intercept[c] + intercept_pupil[p, c];
		}
	}
}

#transformed parameters {
#    vector[2] threshold;
#    threshold[1] = (sr[1]-intercept[1])/slope[1];
#    threshold[2] = (sr[2]-intercept[2])/slope[2];
#}

model {
	int p;
	int c;
    vector[n] mu;

	# Core priors for SR
	sr_alpha ~ normal(0.5, 0.1)T[0, ];
	sr_beta ~ normal(0.1, 0.1)T[0, ];
	sr_cell_pupil[1] ~ gamma(sr_alpha, sr_beta);
	sr_cell_pupil[2] ~ gamma(sr_alpha, sr_beta);

	# Core priors for slope
	s_alpha ~ normal(2, 5)T[0, ];
	s_beta ~ normal(2, 5)T[0, ];
	s_cell_pupil[1] ~ gamma(s_alpha, s_beta);
	s_cell_pupil[2] ~ gamma(s_alpha, s_beta);

	# Core priors for intercept
	i_mean ~ normal(0, 10);
	i_sd ~ normal(0, 5)T[0, ];
	i_cell_pupil[1] ~ normal(i_mean, i_sd);
	i_cell_pupil[2] ~ normal(i_mean, i_sd);

    sigma ~ normal(0, 1);

	for (i in 1:2) {
		sr_pupil[i] ~ normal(0, 1);
		slope_pupil[i] ~ normal(0, 0.1);
		intercept_pupil[i] ~ normal(0, 1);
	}

	for (i in 1:n) {
		p = pupil[i];
		c = cell[i];
		mu[i] = fmax(sr_cell_pupil[p, c], x[i] * s_cell_pupil[p, c] + i_cell_pupil[p, c]);
	}

    y ~ normal(mu, sigma);
}
