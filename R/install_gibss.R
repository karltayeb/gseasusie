#' @importFrom reticulate py_require
.onLoad <- function(...) {
  py_require("gibss")
}

import_gsea_fun <- function() {
  reticulate::py_run_string("
import numpy as np
from scipy import sparse
from gibss.logistic_sparse import fit_logistic_susie2
def logistic_susie_gsea(X, y, **kwargs):
    X_sp = sparse.csr_matrix(X)
    y = np.array(y)
    fit = fit_logistic_susie2(X_sp, y, **kwargs)
    fit2 = fit._asdict()
    return fit2
logistic_susie_gsea")$logistic_susie_gsea
}
