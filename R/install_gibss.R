#' @importFrom reticulate py_install virtualenv_exists virtualenv_remove
#' @export
install_gibss <- function(..., envname = "r-gibss", new_env = identical(envname, "r-gibss")) {
  if (new_env && virtualenv_exists(envname)) {
    virtualenv_remove(envname)
  }
  py_install(packages = "gibss", envname = envname, python_version = ">=3.11", ...)
}

#' @importFrom reticulate use_virtualenv
.onLoad <- function(...) {
  use_virtualenv("r-gibss", require = F)
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
