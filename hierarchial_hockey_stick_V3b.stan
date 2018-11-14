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

transformed parameters {
	vector[n_cells] threshold;
	threshold = (sr-intercept)/slope;
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
	i_mean ~ normal(0, 25);
	i_sd ~ normal(0, 25)T[0, ];

	# Core priors for SR pupil
	srp_mean ~ normal(0, 10);
	srp_sd ~ normal(0, 10)T[0, ];

	# Core priors for intercept pupil
	ip_mean ~ normal(0, 10);
	ip_sd ~ normal(0, 10)T[0, ];

	# Core priors for slope pupil
	sp_mean ~ normal(0, 5);
	sp_sd ~ normal(0, 5)T[0, ];

    sigma ~ normal(0, 10)T[0, ];

	for (i in 1:n_cells) {
		sr[i] ~ gamma(sr_alpha, sr_beta);
		slope[i] ~ gamma(s_alpha, s_beta);
		intercept[i] ~ normal(i_mean, i_sd);

		sr_pupil[i] ~ normal(srp_mean, srp_sd);
		slope_pupil[i] ~ normal(sp_mean, sp_sd);
		intercept_pupil[i] ~ normal(ip_mean, ip_sd);
	}

	for (i in 1:n) {
		p = pupil[i] - 1; # 0 if small, 1 if large
		c = cell[i];
		sr_i = sr[c] + sr_pupil[c] * p;
		intercept_i = intercept[c] + intercept_pupil[c] * p;
		slope_i = slope[c] + slope_pupil[c] * p;
		mu[i] = fmax(sr_i, x[i] * slope_i + intercept_i);
	}

    y ~ normal(mu, sigma);
}
