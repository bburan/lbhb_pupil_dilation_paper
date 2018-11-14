data {
    int<lower=1> n;
    vector[n] x;
    vector[n] y;
}

parameters {
    real<lower=0> slope;
    real<lower=0> sr;
    real intercept;
    
    real sigma;
}

transformed parameters {
    real threshold;
    threshold = (sr-intercept)/slope;
}

model {
    vector[n] mu;
    
    sr ~ gamma(0.5, 0.1);

    slope ~ normal(0.5, 5);
    intercept ~ normal(0, 20);
    sigma ~ normal(0, 1);
    
    mu = x * slope + intercept;
    for (i in 1:n) {
        mu[i] = fmax(sr, mu[i]);
    }
    
    y ~ normal(mu, sigma);
}
