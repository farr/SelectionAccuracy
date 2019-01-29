functions {
  real log_dNdL(real L, real alpha) {
    return -lgamma(1-alpha) - alpha*log(L) - L;
  }
}

data {
  int nobs;
  int nsel;
  int ngen;

  real beta;

  int Fobs[nobs];

  real Lsel[nsel];
  real wtsel[nsel];
}

transformed data {
  real log_wtsel[nsel];

  for (i in 1:nsel) {
    log_wtsel[i] = log(wtsel[i]);
  }
}

parameters {
  real<upper=0.9> alpha; /* Ensure alpha stays regularized. */
  real<lower=0.1, upper=10> LStar;
  real<lower=0> L[nobs];
}

transformed parameters {
  real mu;
  real neff;

  {
    real logps[nsel];
    real logps2[nsel];

    real logp;
    real logp2;

    real nsum;
    real nsum2;

    real sigma2;

    real logLStar = log(LStar);

    for (i in 1:nsel) {
      logps[i] = log_dNdL(Lsel[i]/LStar, alpha) - log_wtsel[i];
      logps2[i] = 2.0*logps[i];
    }

    logp = log_sum_exp(logps) - logLStar;
    logp2 = log_sum_exp(logps2) - 2.0*logLStar;

    nsum = exp(logp-log(ngen));
    nsum2 = exp(logp2 - 2.0*log(ngen));

    mu = nsum;
    sigma2 = nsum2 - mu*mu/ngen;

    neff = mu*mu/sigma2;
  }
}

model {
  alpha ~ normal(0, 1); /* Normal prior */
  LStar ~ lognormal(0, 1); /* */

  for (i in 1:nobs) {
    target += log_dNdL(L[i]/LStar, alpha);
  }
  target += -nobs*log(LStar);

  for (i in 1:nobs) {
    Fobs[i] ~ poisson(beta*L[i]);
  }

  /* Normalization term */
  target += -(nobs+1)*log(mu) + nobs*(3.0 + nobs)/(2*neff);
}

generated quantities {
  real N;

  {
    real muN = nobs/mu*(1.0 + nobs/neff);
    real sigmaN = sqrt(nobs)/mu*(1.0 + 1.5*nobs/neff);

    N = normal_rng(muN, sigmaN);
  }
}
