data {
    int<lower=1> n;
    
    real freq[n];
    int<lower=0> spike_count[n];
    real<lower=0> sample_time[n];
    int<lower=0, upper=1> pupil[n];
    
    real<lower=0> sr;
    real<lower=0> sr_lg;
}

parameters {
    real bf;
    real<lower=0> bw;
    real<lower=0> gain;
    
    real bf_delta;
    real<lower=0> bw_ratio;
    real<lower=0> gain_ratio;
}

model {
    vector[n] lambda;
    
    real gauss;
    real bf_i;
    real bw_i;
    real gain_i;
    real sr_i;
    
    bf ~ normal(7.25, 2.0);
    bw ~ gamma(1, 1);
    gain ~ normal(35, 35)T[0, ];
    
    bf_delta ~ normal(0, 1);
    bw_ratio ~ normal(1, 0.5)T[0, ];
    gain_ratio ~ normal(1, 10);
    
    for (i in 1:n) {
        if (pupil[i] == 0) {
            sr_i = sr;
            bf_i = bf;
            bw_i = bw;
            gain_i = gain;
        } else {
            sr_i = sr_lg;
            bf_i = bf + bf_delta;
            bw_i = bw * bw_ratio;
            gain_i = gain * gain_ratio;
        }
        gauss = exp(-0.5*square((freq[i]-bf_i)/bw_i));
        lambda[i] = sr_i * gain_i * gauss * sample_time[i];
    }
    
    spike_count ~ poisson(lambda);
}