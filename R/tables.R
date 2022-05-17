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
      in_cs = dplyr::if_else(is.na(in_cs), FALSE, in_cs)) %>%
    dplyr::select(geneSet, beta, pip, overlap, geneSetSize, logOddsRatio, nlog10pFishersExact, in_cs, component, fisherRank) %>%
    dplyr::mutate(dplyr::across(!where(is.numeric), as.factor))

  dt %>%
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