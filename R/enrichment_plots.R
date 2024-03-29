#' apply FDR correction to marginal pvalues and label enrichments "depleted" or "enriched"
#' @param tbl a table containing columns corresponding to the other two arguments
#' @param p character name of column with p values
#' @param or character name of column with odds ratios/effect sizes
label_sig_enrichments = function(tbl, p, or){
   tbl %>% dplyr::mutate(
    padj = p.adjust(!!sym(p), 'fdr'),
    result = dplyr::case_when(
      padj < 0.05 & !!sym(or) < 1 ~ 'depleted',
      padj < 0.05 & !!sym(or) > 1 ~ 'enriched',
      TRUE ~ 'not significant'
    )
  )
}

#' make a volcato plot
#' res is a tibble with columns oddsRatio, pFishersExact, in_cs, active_cs
#' @param fit a fit logistic susie
#' @param ora output from `fit_ora`
#' @export
enrichment_volcano = function (fit, ora, or='oddsRatio', p='pFishersExact'){
  sym_p <- sym(p)
  sym_or <- sym(or)
  
  res <- get_gene_set_summary(fit) %>% dplyr::left_join(ora)
  csdat <- get_credible_set_summary(fit) %>% dplyr::left_join(ora) %>% 
    dplyr::filter(in_cs, active_cs)
  res %>% 
    label_sig_enrichments(p=p, or=or) %>% 
    ggplot2::ggplot(aes(x = log10(!!sym_or), y = -log10(!!sym_p), color = result)) +
    ggplot2::geom_point() + 
    ggplot2::geom_point(csdat, mapping = aes(x = log10(!!sym_or), 
      y = -log10(!!sym_p), fill = component), color = "black", 
      pch = 21, size = 5) + ggplot2::scale_color_manual(values = c(
        depleted = "coral",
        enriched = "dodgerblue",
        `not significant` = "grey"))
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

#' @export
enrichment_volcano2 = function(res, p='pFishersExact', or='oddsRatio'){
  res %>% 
    label_sig_enrichments(p=p, or=or) %>%
    ggplot2::ggplot(aes(x=log10(!!sym(or)), y=-log10(!!sym(p)), color=result)) +
    ggplot2::geom_point() +
    ggplot2::geom_point(
      res %>% filter(in_cs, active_cs), 
      mapping=aes(x=log10(!!sym(or)), y=-log10(!!sym(p)), fill=component),
      color='black', pch=21, size=5) +
    ggplot2::scale_color_manual(values = c('depleted' = 'coral',
                                           'enriched' = 'dodgerblue',
                                           'not significant' = 'grey'))
}

#' @export
residual_enrichment_histogram2 = function(res){
  res %>%
    dplyr::select(geneSet, pval_marginal, pval_residual) %>%
    tidyr::pivot_longer(dplyr::starts_with('pval'), values_to = 'pval') %>%
    ggplot2::ggplot(aes(x=pval)) +
      ggplot2::geom_histogram() + ggplot2::facet_wrap(vars(name))
}