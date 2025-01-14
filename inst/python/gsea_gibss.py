from dataclasses import dataclass
import numpy as np
from gibss.logisticprofile import logistic_susie, logistic_ser_hermite

@dataclass
class GSEARes:
  alpha: np.ndarray
  effects: np.ndarray
  log_bfs: np.ndarray
  
get_alpha = lambda fit: np.array([c.alpha for c in fit.components])
get_beta = lambda fit: np.array([c.fits.beta for c in fit.components])
get_lbf = lambda fit: np.array([c.lbf_ser for c in fit.components])

def gsea_gibss(X, y, L=5, prior_variance = 1, maxiter = 10, tol=1e-3):
    # TODO: check types
    X = np.array(X).astype(float)
    y = np.array(y).astype(float)
    fit = logistic_susie(
        X, y,
        L=int(L), method='hermite',
        serkwargs=dict(m=1, prior_variance=1.0)
    )
    return GSEARes(get_alpha(fit), get_beta(fit), get_lbf(fit))
