color_sign = function(value) {
      if (value > 0) {
        color <- "#008000"
      } else if (value < 0) {
        color <- "#e00000"
      } else {
        color <- "#777"
      }
      list(color = color)
}


#' @export
interactive_table = function(fit, ora){
  res <- gseasusie:::get_gene_set_summary(fit) %>%
  dplyr::left_join(ora)

  csdat <- gseasusie:::get_credible_set_summary(fit) %>%
    dplyr::left_join(ora) %>%
    dplyr::filter(in_cs, active_cs) %>%
    dplyr::select(geneSet, component, in_cs) %>%
    distinct()
  
  dt <- res %>% 
    dplyr::filter(overlap > 0) %>%
    dplyr::mutate(
      logOddsRatio = log(oddsRatio),
      nlog10pFishersExact = -log10(pFishersExact)
    ) %>%
    dplyr::left_join(csdat) %>%
    dplyr::arrange(dplyr::desc(nlog10pFishersExact)) %>%
    dplyr::mutate(
      fisherRank = dplyr::row_number(),
      in_cs = dplyr::if_else(is.na(in_cs), FALSE, in_cs)) %>%
    dplyr::select(geneSet, beta, pip, overlap, geneSetSize, logOddsRatio, nlog10pFishersExact, in_cs, component, fisherRank) %>%
    dplyr::mutate(dplyr::across(!where(is.numeric) , as.factor))

  dt %>%
    dplyr::select(-c(in_cs)) %>%
    reactable::reactable(
      filterable=TRUE,
      minRows=10,
      columns = list(
        pip = reactable::colDef(format = reactable::colFormat(digits = 3)),
        logOddsRatio = reactable::colDef(style= function(value){color_sign(value)},
                                         format = reactable::colFormat(digits = 3)),
        beta = reactable::colDef(style= function(value){color_sign(value)},
                                 format = reactable::colFormat(digits = 3)),
        nlog10pFishersExact = reactable::colDef(format = reactable::colFormat(digits = 3))
      ),
      rowStyle = function(index){
        if(dt$in_cs[index] == TRUE){
          list(background = "#e5f5e0")
        }
      },
      defaultSorted = list(nlog10pFishersExact='desc')
    )
}


#' @export
interactive_table2 = function(res){
  dt <- res %>% 
    dplyr::filter(overlap > 0) %>%
    dplyr::mutate(
      logOddsRatio = log(oddsRatio),
      nlog10pFishersExact = -log10(pFishersExact)
    ) %>%
    dplyr::arrange(dplyr::desc(nlog10pFishersExact)) %>%
    dplyr::mutate(
      fisherRank = dplyr::row_number(),
      in_active_cs = dplyr::if_else(is.na(in_cs), FALSE, in_cs & active_cs)) %>%
    dplyr::select(geneSet, description, in_active_cs, beta, pip, overlap, geneSetSize, logOddsRatio, nlog10pFishersExact, component, fisherRank) %>%
    dplyr::mutate(dplyr::across(!where(is.numeric), as.factor))

  dt %>%
    reactable::reactable(
      filterable=TRUE,
      minRows=20,
      columns = list(
        pip = reactable::colDef(format = reactable::colFormat(digits = 3)),
        logOddsRatio = reactable::colDef(style= function(value){color_sign(value)},
                                         format = reactable::colFormat(digits = 3)),
        beta = reactable::colDef(style= function(value){color_sign(value)},
                                 format = reactable::colFormat(digits = 3)),
        nlog10pFishersExact = reactable::colDef(format = reactable::colFormat(digits = 3))
      ),
      rowStyle = function(index){
        if(dt$in_active_cs[index] == TRUE){
          list(background = "#e5f5e0")
        }
      },
      defaultSorted = list(nlog10pFishersExact='desc')
    )
}

pack_group = function(tbl){
    components <- tbl$component
    unique.components <- unique(components)
    start <- match(unique.components, components)
    end <- c(tail(start, -1) - 1, length(components))
    res <- tbl %>% dplyr::select(-c(component)) %>% kableExtra::kbl()
    for(i in 1:length(unique.components)){
      res <- kableExtra::pack_rows(res, unique.components[i], start[i], end[i])
    }
    return(res)
}

#' Report credible set based summary of SuSiE
#' @export
static_table = function(fit, ora){
  res <- gseasusie:::get_gene_set_summary(fit) %>%
    dplyr::left_join(ora)

  csdat <- gseasusie:::get_credible_set_summary(fit) %>%
    dplyr::left_join(ora) %>%
    dplyr::filter(in_cs, active_cs) %>%
    dplyr::select(geneSet, description, component, in_cs, alpha) %>%
    distinct()
  
  dt <- res %>% 
    dplyr::filter(overlap > 0) %>%
    dplyr::mutate(
      logOddsRatio = log(oddsRatio),
      nlog10pFishersExact = -log10(pFishersExact)
    ) %>%
    dplyr::left_join(csdat) %>%
    dplyr::arrange(dplyr::desc(nlog10pFishersExact)) %>%
    dplyr::mutate(
      fisherRank = dplyr::row_number(),
      in_cs = dplyr::if_else(is.na(in_cs), FALSE, in_cs)) %>%
    dplyr::filter(in_cs) %>%
    dplyr::select(tidyselect::any_of(
      c("geneSet", "description", "beta", "alpha", "pip", "overlap", "geneSetSize", "logOddsRatio", "nlog10pFishersExact", "in_cs", "component", "fisherRank"))) %>%
    dplyr::mutate(dplyr::across(!where(is.numeric) , as.factor))

  dt %>%
    dplyr::select(
      component, geneSet, description, geneSetSize, overlap,
      logOddsRatio, beta,
      alpha, pip, nlog10pFishersExact, fisherRank) %>%
    dplyr::mutate_if(is.numeric, funs(as.character(signif(., 3)))) %>%
    pack_group %>%
    kableExtra::column_spec(c(7), color=ifelse(dt$beta > 0, 'green', 'red')) %>%
    kableExtra::kable_styling()
}