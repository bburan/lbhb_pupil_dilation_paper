data {
    int<lower=1> n;
    vector[n] evoked_level;
	real<lower=0> evoked_time[n];
    int<lower=0> evoked_count[n];
	int<lower=0> sr_count;
	real<lower=0> sr_time;
}


parameters {
    real slope;
    real<lower=0> sr;
    real threshold;
}


model {
	real lambda[n];

    sr ~ gamma(0.5, 0.1);
    slope ~ normal(0.1, 1);
	threshold ~ normal(40, 20);

	for (i in 1:n) {
		if (evoked_level[i] <= threshold) {
			lambda[i] = sr * evoked_time[i];
		} else {
			lambda[i] = (slope*(evoked_level[i]-threshold) + sr) * evoked_time[i];
		}
	}
    evoked_count ~ poisson(lambda);
	sr_count ~ poisson(sr * sr_time);
}
