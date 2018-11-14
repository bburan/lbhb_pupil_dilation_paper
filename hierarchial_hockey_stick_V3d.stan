data {
    int<lower=1> n;						# number of rows
	int<lower=1> n_cells;				# number of cells
    vector[n] x;						# level
    vector[n] y;						# spike rate
	int<lower=1, upper=2> pupil[n];		# pupil index
	int<lower=1> cell[n];				# cell index
}

parameters {
	real t_mean;
	real<lower=0> t_sd;

	real sr_mean;
	real<lower=0> sr_sd;

	real s_mean;
	real<lower=0> s_sd;

	real<lower=0> sr[n_cells];
	real<lower=0> slope[n_cells];
	real threshold[n_cells];

	real sp_mean;
	real<lower=0> sp_sd;

	real tp_mean;
	real<lower=0> tp_sd;

	real srp_mean;
	real<lower=0> srp_sd;

	vector[n_cells] sr_pupil;
	vector[n_cells] slope_pupil;
	vector[n_cells] threshold_pupil;
	
    real sigma;
}

	
model {
	real sr_i;
	real threshold_i;
	real slope_i;
	real intercept_i;

	int p;
	int c;
    vector[n] mu;

	# Core priors for SR
	sr_mean ~ normal(10, 10)T[0, ];
	sr_sd ~ normal(0, 10)T[0, ];

	# Core priors for slope
	s_mean ~ normal(2, 5)T[0, ];
	s_sd ~ normal(2, 5)T[0, ];

	# Core priors for threshold
	t_mean ~ normal(50, 25);
	t_sd ~ normal(0, 25)T[0, ];

	# Core priors for SR pupil
	srp_mean ~ normal(0, 10);
	srp_sd ~ normal(0, 10)T[0, ];

	# Core priors threshold pupil
	tp_mean ~ normal(0, 10);
	tp_sd ~ normal(0, 10)T[0, ];

	# Core priors for slope pupil
	sp_mean ~ normal(0, 5);
	sp_sd ~ normal(0, 5)T[0, ];

    sigma ~ normal(0, 1)T[0, ];

	for (i in 1:n_cells) {
		sr[i] ~ normal(sr_mean, sr_sd);
		slope[i] ~ normal(s_mean, s_sd);
		threshold[i] ~ normal(t_mean, t_sd);

		sr_pupil[i] ~ normal(srp_mean, srp_sd);
		slope_pupil[i] ~ normal(sp_mean, sp_sd);
		threshold_pupil[i] ~ normal(tp_mean, tp_sd);
	}

	for (i in 1:n) {
		p = pupil[i] - 1; # 0 if small, 1 if large
		c = cell[i];

		sr_i = sr[c] + sr_pupil[c] * p;
		threshold_i = threshold[c] + threshold_pupil[c] * p;
		slope_i = slope[c] + slope_pupil[c] * p;

		intercept_i = sr_i - threshold_i * slope_i;

		mu[i] = fmax(sr_i, x[i] * slope_i + intercept_i);
	}

    y ~ normal(mu, sigma);
}
