# We make a nested table to help navigate results

make_cs_tbl_single <- function(fit, l) {
  cs <- fit$fit$credible_sets[[l]]
  tidyr::as_tibble(cs) %>%
    dplyr::mutate(
      geneSetIdx = cs + 1,
      component = paste0("L", l),
      lbf_ser = fit$fit$lbf_ser[[l]]
    ) %>%
    dplyr::select(-cs) %>%
    dplyr::left_join(fit$data$geneSets) %>%
    dplyr::group_by(component, geneSet) %>%
    dplyr::mutate(
      geneSetSize = n(),
      propInList = mean(geneInList)
    ) %>%
    dplyr::ungroup()
}

make_cs_tbl <- function(fit, min_lbf_ser = log(10.)) {
  # include components with large enough lbf_ser
  include_components <- which(fit$fit$lbf_ser > min_lbf_ser)
  purrr::map_dfr(include_components, ~ make_cs_tbl_single(fit, .x))
}

make_component_tbl_single <- function(fit, l) {
  # component, gene set, alpha, beta, lbf, prior_variance
  with(fit$fit, tibble::tibble(
    component = glue::glue("L{l}"),
    geneSet = fit$data$geneSetMapping$geneSet,
    alpha = alpha[l, ],
    beta = beta[l, ],
    lbf = lbf[l, ],
    prior_variance = prior_variance[l]
  ))
}

make_component_tbl <- function(fit, min_lbf_ser = log(10)) {
  include_components <- which(fit$fit$lbf_ser >= min_lbf_ser)
  purrr::map_dfr(include_components, ~ make_component_tbl_single(fit, .x))
}

#' @export
make_cs_tbl_nested <- function(fit) {
  more_go_info <- AnnotationDbi::select(GO.db::GO.db,
    keys = unique(fit$data$geneSets$geneSet),
    columns = c("TERM", "DEFINITION"),
    keytype = "GOID"
  ) %>%
    as_tibble() %>%
    dplyr::mutate(geneSet = GOID)

  more_gene_info <- AnnotationDbi::select(org.Hs.eg.db::org.Hs.eg.db,
    keys = unique(fit$data$geneMapping$gene),
    columns = c("SYMBOL", "GENETYPE", "GENENAME"),
    keytype = "ENTREZID"
  ) %>%
    as_tibble() %>%
    dplyr::mutate(gene = ENTREZID)

  gene_columns <- c("ENTREZID", "SYMBOL", "GENENAME", "geneInList")
  gene_set_columns <- c("geneSet", "TERM", "DEFINITION", "alpha", "beta", "lbf", "geneSetSize", "propInList")
  component_columns <- c("component", "size", "coverage", "target_coverage", "lbf_ser", "prior_variance")

  all_columns <- c(component_columns, gene_set_columns, gene_columns)

  message("building nested credible set table")
  fit %>%
    make_cs_tbl() %>%
    left_join(make_component_tbl(fit)) %>% # add effect estimates, etc.
    left_join(more_go_info) %>%
    left_join(more_gene_info) %>%
    dplyr::select(all_columns) %>%
    # nest gene level data
    tidyr::nest(.by = c(component_columns, gene_set_columns), .key = "details") %>%
    # nest gene set level data
    tidyr::nest(.by = component_columns, .key = "details")
}

make_get_details <- function(tbl) {
  if ("details" %in% names(tbl)) {
    get_details <- function(index) {
      deets <- tbl[index, ]
      htmltools::div(
        style = "padding: 1rem",
        reactable::reactable(
          dplyr::select(deets$details[[1]], -any_of("detials")),
          details = make_get_details(deets$details[[1]]),
          outlined = TRUE,
          defaultColDef = reactable::colDef(format = reactable::colFormat(digits = 3))
        )
      )
    }
  } else {
    get_details <- NULL
  }
  return(get_details)
}

#' @export
nested_reactable <- function(nested_tbl) {
  reactable::reactable(
    dplyr::select(nested_tbl, -tidyselect::any_of("details")),
    detail = make_get_details(nested_tbl)
  )
}
