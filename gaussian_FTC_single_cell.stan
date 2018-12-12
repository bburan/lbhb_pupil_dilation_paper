data {
    int<lower=1> n;
    vector[n] freq;
    int<lower=0> spike_count[n];
    vector[n] sample_time;
    vector[n] pupil;
    int<lower=0> spont_count[2];
    real<lower=0> spont_time[2];
}
parameters {
    real<lower=0> bf;
    real bf_pupil_delta;
    
    real<lower=0> bw;
    real bw_pupil_delta;
    
    real gain;
    real gain_pupil_delta;
    
    real<lower=0> offset;
    real offset_pupil_delta;
}

model {
    vector[n] lambda;
    vector[2] lambda_sr;
    
    real bf_i;
    real bw_i;
    real gain_i;
    real offset_i;
    
    bf ~ normal(7.25, 2.0);
    bf_pupil_delta ~ normal(0, 1);
    bw ~ normal(1, 1)T[0, ];
    bw_pupil_delta ~ normal(0, 0.5);
    offset ~ gamma(0.65, 0.07);
    offset_pupil_delta ~ normal(0, 2);
    gain ~ normal(35, 35)T[-offset, ];
    gain_pupil_delta ~ normal(0, 10);
    
    for (i in 1:n) {
        bf_i = bf + bf_pupil_delta * pupil[i];
        bw_i = bw + bw_pupil_delta * pupil[i];
        offset_i = offset + offset_pupil_delta * pupil[i];
        gain_i = gain + gain_pupil_delta * pupil[i];
        
        lambda[i] = exp(-0.5*square((freq[i]-bf_i)/bw_i));
        lambda[i] = offset_i + gain_i * lambda[i];
        lambda[i] = lambda[i] * sample_time[i];
    }
    
    lambda_sr[1] = offset * spont_time[1];
    lambda_sr[2] = (offset + offset_pupil_delta) * spont_time[2];
    spike_count ~ poisson(lambda);
    spont_count ~ poisson(lambda_sr);
}
