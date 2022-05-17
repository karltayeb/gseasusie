label_sig_enrichments = function(tbl){
   tbl %>% dplyr::mutate(
    padj = p.adjust(pFishersExact),
    result = dplyr::case_when(
      padj < 0.05 & oddsRatio < 1 ~ 'depleted',
      padj < 0.05 & oddsRatio > 1 ~ 'enriched',
      TRUE ~ 'not significant'
    )
  )
}

#' make a volcato plot
#' res is a tibble with columns oddsRatio, pFishersExact, in_cs, active_cs
#' @param fit a fit logistic susie
#' @param ora output from `fit_ora`
#' @export
enrichment_volcano = function(fit, ora){
  res <- get_gene_set_summary(fit) %>%
    dplyr::left_join(ora)
  
  csdat <- get_credible_set_summary(fit) %>%
    dplyr::left_join(ora) %>%
    dplyr::filter(in_cs, active_cs)
  
  res %>% 
    label_sig_enrichments() %>%
    ggplot2::ggplot(aes(x=log10(oddsRatio), y=-log10(pFishersExact), color=result)) +
    ggplot2::geom_point() +
    ggplot2::geom_point(
      csdat, 
      mapping=aes(x=log10(oddsRatio), y=-log10(pFishersExact), fill=component),
      color='black', pch=21, size=5) +
    ggplot2::scale_color_manual(values = c('depleted' = 'coral',
                                           'enriched' = 'dodgerblue',
                                           'not significant' = 'grey'))
}

#' @export
residual_enrichment_histogram = function(marginal_regression, residual_regression){
  plotdat <- rbind(
    marginal_regression %>%
      dplyr::select(geneSet, pval) %>% 
      dplyr::mutate(model='marginal'),
    residual_regression %>% 
      dplyr::select(geneSet, pval) %>%
      dplyr::mutate(model='residual')
  )
  
  plotdat %>% ggplot(aes(x=pval)) +
    ggplot2::geom_histogram() + ggplot2::facet_wrap(vars(model))
}