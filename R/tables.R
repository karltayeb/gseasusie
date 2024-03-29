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
  # get credible sets
  res <- gseasusie:::get_gene_set_summary(fit) %>%
    dplyr::left_join(ora)
  csdat <- gseasusie:::get_credible_set_summary(fit) %>%
    dplyr::left_join(ora) %>%
    dplyr::filter(in_cs, active_cs) %>%
    dplyr::select(geneSet, description, component, in_cs, alpha, conditional_beta) %>%
    distinct()
  
  # manipulate table
  columns <- c(
    'geneSet', 'description', 'geneSetSize', 'overlap',
    'log2OR', 'effect', 'alpha', 'pip', 'nlog10pFishersExact', 'fisherRank',
    'component'
  )
  color_columns <- which(columns %in% c('log2OR', 'effect'))
  dt <- res %>% 
    dplyr::filter(overlap > 0) %>%
    dplyr::mutate(
      log2OR = log2(oddsRatio),
      nlog10pFishersExact = -log10(pFishersExact)
    ) %>%
    dplyr::left_join(csdat) %>%
    dplyr::arrange(dplyr::desc(nlog10pFishersExact)) %>%
    dplyr::mutate(
      fisherRank = dplyr::row_number(),
      in_cs = dplyr::if_else(is.na(in_cs), FALSE, in_cs),
      effect = conditional_beta * log2(exp(1))
    ) %>%
    dplyr::filter(in_cs) %>%
    dplyr::select(columns) %>%
    dplyr::mutate(dplyr::across(!where(is.numeric) , as.factor)) %>%
    mutate(
      component = reorder(factor(component), fisherRank, FUN=min) 
      # sorts components
    ) %>%
    dplyr::arrange(component) %>%
    dplyr::mutate_if(is.numeric, funs(as.character(signif(., 3))))

  # display table
  dt %>%
    pack_group %>%
    kableExtra::column_spec(
      color_columns, color=ifelse(dt$effect > 0, 'green', 'red')) %>%
    kableExtra::kable_styling()
}

#' @export
static_table2 = function(res){
  require(kableExtra)
  tbl_filtered <-
    res %>%
    arrange(pFishersExact) %>%
    mutate(fisherRank = row_number()) %>%
    filter(in_cs, active_cs) %>%
    group_by(component) %>%
    arrange(component, desc(alpha)) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(logOddsRatio = log10(oddsRatio))

  tbl_filtered %>%
    dplyr::select(
      component, geneSet, description, geneSetSize, overlap,
      logOddsRatio, conditional_beta, conditional_beta_se,
      alpha, pip, pFishersExact, fisherRank) %>%
    dplyr::mutate_if(is.numeric, funs(as.character(signif(., 3)))) %>%
    pack_group %>%
    column_spec(c(4), color=ifelse(tbl_filtered$beta > 0, 'green', 'red')) %>%
    kableExtra::kable_styling()
}
