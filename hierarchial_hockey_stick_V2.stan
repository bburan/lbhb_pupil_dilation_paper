data {
    int<lower=1> n;
    vector[n] x;
    vector[n] y;
	int<lower=1, upper=2> pupil[n];
}

parameters {
	real<lower=0> alpha;
	real<lower=0> beta;
	real i_mean;
	real<lower=0> i_sd;
	real s_mean;
	real<lower=0> s_sd;
	vector[2] sr;
	vector[2] slope;
	vector[2] intercept;
    real sigma;
}

transformed parameters {
    vector[2] threshold;
    threshold[1] = (sr[1]-intercept[1])/slope[1];
    threshold[2] = (sr[2]-intercept[2])/slope[2];
}

model {
	int p;
    vector[n] mu;

	alpha ~ normal(0.5, 0.1);
	beta ~ normal(0.1, 0.1);

	i_mean ~ normal(0, 10);
	i_sd ~ normal(0, 5)T[0, ];
	s_mean ~ normal(0.5, 5);
	s_sd ~ normal(0, 1)T[0, ];

	sr[1] ~ gamma(alpha, beta);
	sr[2] ~ gamma(alpha, beta);
	intercept[1] ~ normal(i_mean, i_sd);
	intercept[2] ~ normal(i_mean, i_sd);
	slope[1] ~ normal(s_mean, s_sd);
	slope[2] ~ normal(s_mean, s_sd);

    sigma ~ normal(0, 1);
    
    for (i in 1:n) {
		p = pupil[i];
		mu[i] = fmax(sr[p], x[i] * slope[p] + intercept[p]);
    }
    
    y ~ normal(mu, sigma);
}
