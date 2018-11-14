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

	real sp_mean;
	real<lower=0> sp_sd;
	real ip_mean;
	real<lower=0> ip_sd;
	real srp_mean;
	real<lower=0> srp_sd;

	vector[n_cells] sr_pupil;
	vector[n_cells] slope_pupil;
	vector[n_cells] intercept_pupil;
	
    real sigma;
}

model {
	real sr_i;
	real intercept_i;
	real slope_i;

	int p;
	int c;
    vector[n] mu;

	# Core priors for SR
	sr_alpha ~ normal(0.5, 0.1)T[0, ];
	sr_beta ~ normal(0.1, 0.1)T[0, ];

	# Core priors for slope
	s_alpha ~ normal(2, 5)T[0, ];
	s_beta ~ normal(2, 5)T[0, ];

	# Core priors for intercept
	i_mean ~ normal(0, 10);
	i_sd ~ normal(0, 5)T[0, ];

	# Core priors for SR pupil
	srp_mean ~ normal(0, 10);
	srp_sd ~ normal(0, 5)T[0, ];

	# Core priors for intercept pupil
	ip_mean ~ normal(0, 10);
	ip_sd ~ normal(0, 5)T[0, ];

	# Core priors for slope pupil
	sp_mean ~ normal(0, 1);
	sp_sd ~ normal(0, 1)T[0, ];

    sigma ~ normal(0, 1);

	for (i in 1:n_cells) {
		sr[i] ~ gamma(sr_alpha, sr_beta);
		slope[i] ~ gamma(s_alpha, s_beta);
		intercept[i] ~ normal(i_mean, i_sd);

		sr_pupil[1, i] ~ normal(srp_mean, srp_sd);
		sr_pupil[2, i] ~ normal(srp_mean, srp_sd);

		intercept_pupil[1, i] ~ normal(ip_mean, ip_sd);
		intercept_pupil[2, i] ~ normal(ip_mean, ip_sd);

		slope_pupil[1, i] ~ normal(sp_mean, sp_sd);
		slope_pupil[2, i] ~ normal(sp_mean, sp_sd);
	}

	for (i in 1:n) {
		p = pupil[i];
		c = cell[i];
		sr_i = sr[c] + sr_pupil[p, c];
		intercept_i = intercept[c] + intercept_pupil[p, c];
		slope_i = slope[c] + slope_pupil[p, c];
		mu[i] = fmax(sr_i, x[i] * slope_i + intercept_i);
	}

    y ~ normal(mu, sigma);
}
