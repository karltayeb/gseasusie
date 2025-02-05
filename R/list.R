#' Construct gene set matrix out of gene set table
#' @param tbl a tibble that has at least two columns: `geneSet` and `gene`
#' @returns a sparse gene_set x genes membership matrix where
#'          gene set and gene names stored in `colnames` and `rownames` respectively.
#' @export
construct_geneset_matrix <- function(db) {
  tbl <- db %>%
    dplyr::mutate(
      geneSetIdx = vctrs::vec_group_id(geneSet),
      geneIdx = vctrs::vec_group_id(gene)
    )
  geneMapping <- tbl %>%
    dplyr::select(c(gene, geneIdx)) %>%
    dplyr::distinct()
  geneSetMapping <- tbl %>%
    dplyr::select(c(geneSet, geneSetIdx)) %>%
    dplyr::distinct()
  X <- tbl %>%
    {
      Matrix::sparseMatrix(.$geneIdx, .$geneSetIdx, x = 1.)
    } %>%
    Matrix::t()
  rownames(X) <- geneSetMapping$geneSet
  colnames(X) <- geneMapping$gene
  return(list(X = X, geneSetMapping = geneSetMapping, geneMapping = geneMapping, geneSets = tbl))
}

#' Convert list of gene aliases to entrez ids
#' if 1:many mapping occurs pick the first entrez id listed in the AnnotationDbi organism database map
#' @export
convert_alias_to_entrez <- function(list, id_type = "alias", organism = "hsapiens") {
  map2entrez <- switch(tolower(organism),
    hsapiens = as.list(org.Hs.eg.db::org.Hs.egALIAS2EG),
    mmusculus = as.list(org.Mm.eg.db::org.Mm.egALIAS2EG),
    stop("Unsupported organism")
  )
  list(
    gene_list = unlist(purrr::map(data$gene_list, ~ map2entrez[[.x]][1])),
    gene_background = unlist(purrr::map(data$gene_background, ~ map2entrez[[.x]][1]))
  )
}

#' make X and y for logistic SuSiE
#' @export
prepare_data_webgestalt <- function(list, background, db) {
  if (is.character(db)) {
    db <- WebGestaltR::loadGeneSet(enrichDatabase = db)$geneSet
  }
  # get genes included in analysis, record which are excluded
  genes_in_analysis <- intersect(background, unique(db$gene))
  genes_excluded <- setdiff(background, genes_in_analysis)
  # construct X
  X <- db %>%
    dplyr::filter(gene %in% genes_in_analysis) %>%
    construct_geneset_matrix()
  X$geneSets <- X$geneSets %>%
    dplyr::mutate(geneInList = gene %in% list)
  # construct y
  y <- as.numeric(colnames(X$X) %in% list)
  return(c(X, list(y = y, included = genes_in_analysis, excluded = genes_excluded)))
}

#' Fit logistic SuSiE using gene sets available from WebGestaltR
#' @param list a list of interesting genes
#' @param background a list of background genes
#' @param enrichDatabase a database of genesets available in `WebGestaltR`, see `WebGestaltR::listGeneSet()`
#' @returns a list with three items `$fit` `$data` and `$time`.
#' @export
fit_gsea_susie_webgestalt <- function(list, background, enrichDatabase = "geneontology_Biological_Process", organism = "hsapiens", ...) {
  # load
  logistic_susie_gsea <- import_gsea_fun()
  gsdb <- WebGestaltR::loadGeneSet(enrichDatabase = enrichDatabase, organism = organism)
  tictoc::tic()
  fit <- gsdb$geneSet %>%
    prepare_data_webgestalt(list, background, .) %>%
    {
      list(fit = logistic_susie_gsea(.$X, .$y, ...), data = .)
    }
  time <- tictoc::toc()
  fit$time <- with(time, unname(toc - tic))
  fit$organism <- organism
  fit$enrichDatabase <- enrichDatabase
  return(fit)
}
