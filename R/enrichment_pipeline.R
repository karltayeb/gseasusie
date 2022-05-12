#' Pepare gene set matrix and binary gene list
#' @param gs a list containing gene set matrix `X`
#'    with genes in `rownames` and gene set names in `colnames
#' @param dat a vector of statistics to threshold hold,
#'    names must match columns of `X`
#' @param threshold to binarize `dat`
#' @export
prep_binary_data = function(gs, dat, thresh, .sign=c(1, -1)) {
  # get common genes as background
  gs.genes <- rownames(gs$X)

  # dplyr::filter gene list
  y <- dat %>%
    dplyr::filter(ENTREZID %in% gs.genes, !is.na(threshold.on)) %>%
    dplyr::mutate(threshold.on = dplyr::if_else(sign(beta) %in% .sign, threshold.on, 1)) %>%
    dplyr::mutate(geneList = as.integer(threshold.on < thresh)) %>%
    dplyr::select(ENTREZID, geneList) %>%
    dplyr::distinct(across(ENTREZID), .keep_all = T) %>%
    tibble2namedlist()

  # dplyr::filter gene sets
  y.genes <- names(y)
  test.genes <- intersect(gs.genes, y.genes)
  X <- gs$X[test.genes,]
  bad.cols <- BiocGenerics::colSums(X)
  bad.cols <- (bad.cols == 0) | bad.cols == length(test.genes)
  X <- X[, which(!bad.cols)]

  # reorder genes in y to match X order
  y <- y[rownames(X)]
  return(list(y=y, X=X))
}

#' @export
prep_sumstat_data = function(gs, dat) {
  # get common genes as background
  gs.genes <- rownames(gs$X)

  # dplyr::filter gene list
  subset_dat <- dat %>%
    dplyr::filter(ENTREZID %in% gs.genes, !is.na(beta), !is.na(se)) %>%
    dplyr::distinct(across(ENTREZID), .keep_all = T)

  beta <- subset_dat %>%
    dplyr::select(ENTREZID, beta) %>%
    tibble2namedlist()

  se <- subset_dat %>%
    dplyr::select(ENTREZID, se) %>%
    tibble2namedlist()

  # dplyr::filter gene sets
  y.genes <- names(beta)
  test.genes <- intersect(gs.genes, y.genes)
  X <- gs$X[test.genes,]
  bad.cols <- BiocGenerics::colSums(X)
  bad.cols <- (bad.cols == 0) | bad.cols == length(test.genes)
  X <- X[, which(!bad.cols)]

  # reorder genes in y to match X order
  beta <- beta[rownames(X)]
  se <- se[rownames(X)]
  return(list(beta=beta, se=se, X=X))
}

#' @export
do_logistic_susie = function(experiment,
                             db,
                             thresh,
                             genesets,
                             data,
                             susie.args=NULL,
                             .sign=c(1, -1)) {
  cat(paste0('Fitting logistic susie...',
    '\n\tExperiment = ', experiment,
    '\n\tDatabase = ', db,
    '\n\tthresh = ', thresh))

  gs <- genesets[[db]]
  dat <- data[[experiment]]
  u <- prep_binary_data(gs, dat, thresh, .sign)  # subset to common genes

  if(is.null(susie.args)){  # default SuSiE args
    susie.args = list(
      L=10, init.intercept=0, verbose=1, maxit=500, standardize=FALSE)
  }
  vb.fit <- exec(fit_logistic_susie, u$X, u$y, !!!susie.args)
  res = tibble(
    experiment=experiment,
    db=db,
    thresh=thresh,
    fit=list(vb.fit),
    susie.args = list(susie.args)
  )
  return(res)
}

#' @export
do_linear_susie = function(experiment,
                           db,
                           thresh,
                           genesets,
                           data,
                           susie.args=NULL,
                           .sign=c(1, -1)) {
  cat(paste0('Fitting linear susie...',
    '\n\tExperiment = ', experiment,
    '\n\tDatabase = ', db,
    '\n\tthresh = ', thresh))

  gs <- genesets[[db]]
  dat <- data[[experiment]]
  u <- prep_binary_data(gs, dat, thresh, .sign)  # subset to common genes

  if(is.null(susie.args)){  # default SuSiE args
    susie.args = list(L=10, standardize=FALSE)
  }

  vb.fit <- exec(susieR::susie, u$X, u$y, !!!susie.args)
  res = tibble(
    experiment=experiment,
    db=db,
    thresh=thresh,
    fit=list(vb.fit),
    susie.args = list(susie.args)
  )
  return(res)
}

#' @export
do_tccm_point_normal_susie = function(experiment,
                             db,
                             genesets,
                             data,
                             susie.args=NULL) {
  cat(paste0(
    'Fitting point-normal susie...',
    '\n\tExperiment = ', experiment,
    '\n\tDatabase = ', db))
  gs <- genesets[[db]]
  dat <- data[[experiment]]
  u <- prep_sumstat_data(gs, dat)  # subset to common genes

  if(is.null(susie.args)){  # default SuSiE args
    susie.args = list(L=10, verbose=T, maxit=500)
  }
  fit <- exec(fit_tccm_point_normal_susie, u$beta, u$se, u$X, !!!susie.args)
  res = tibble(
    experiment=experiment,
    db=db,
    fit=list(fit),
    susie.args = list(susie.args)
  )
  return(res)
}

#' @export
do_ora = function(experiment,
                  db,
                  thresh,
                  genesets,
                  data,
                  .sign=c(1,-1)){
  gs <- genesets[[db]]
  dat <- data[[experiment]]
  u <- prep_binary_data(gs, dat, thresh, .sign)  # subset to common genes
  ora <- fit_ora(u$X, u$y) 
  # add description
  ora <- ora %>%
    dplyr::left_join(gs$geneSet$geneSetDes)
  res = tibble(
    experiment=experiment,
    db=db,
    thresh=thresh,
    ora=list(ora)
  )

  return(res)
}

get_credible_set_summary = function(res){
  #' report top 50 elements in cs
  beta <- t(res$mu) %>%
    data.frame() %>%
    rownames_to_column(var='geneSet') %>%
    rename_with(~str_replace(., 'X', 'L')) %>%
    dplyr::rename(L1 = 2) %>%  # rename deals with L=1 case
    pivot_longer(starts_with('L'), names_to='component', values_to = 'beta')
  se <- t(sqrt(res$mu2 - res$mu^2)) %>%
     data.frame() %>%
      rownames_to_column(var='geneSet') %>%
      rename_with(~str_replace(., 'X', 'L')) %>%
      dplyr::rename(L1 = 2) %>%  # rename deals with L=1 case
      pivot_longer(starts_with('L'), names_to='component', values_to = 'beta.se')

  credible.set.summary <- t(res$alpha) %>%
    data.frame() %>%
    rownames_to_column(var='geneSet') %>%
    rename_with(~str_replace(., 'X', 'L')) %>%
    dplyr::rename(L1 = 2) %>%  # rename deals with L=1 case
    pivot_longer(starts_with('L'), names_to='component', values_to = 'alpha') %>%
    left_join(beta) %>%
    left_join(se) %>%
    arrange(component, desc(alpha)) %>%
    dplyr::group_by(component) %>%
    dplyr::filter(row_number() < 50) %>%
    dplyr::mutate(alpha_rank = row_number(), cumalpha = c(0, head(cumsum(alpha), -1))) %>%
    dplyr::mutate(in_cs = cumalpha < 0.95) %>%
    dplyr::mutate(active_cs = component %in% names(res$sets$cs))
  return(credible.set.summary)
}

get_gene_set_summary = function(res){
  #' map each gene set to the component with top alpha
  #' report pip
  res$pip %>%
    as_tibble(rownames='geneSet') %>%
    dplyr::rename(pip=value) %>%
    dplyr::mutate(beta=colSums(res$alpha * res$mu))
}

pack_group = function(tbl){
    components <- tbl$component
    unique.components <- unique(components)
    start <- match(unique.components, components)
    end <- c(tail(start, -1) - 1, length(components))
    res <- tbl %>% dplyr::select(-c(component)) %>% kbl()
    for(i in 1:length(unique.components)){
      res <- pack_rows(res, unique.components[i], start[i], end[i])
    }
    return(res)
}


label_sig_enrichments = function(tbl){
   tbl %>% dplyr::mutate(
    padj = p.adjust(pFishersExact),
    result = case_when(
      padj < 0.05 & oddsRatio < 1 ~ 'depleted',
      padj < 0.05 & oddsRatio > 1 ~ 'enriched',
      TRUE ~ 'not significant'
    )
  )
}

do.volcano = function(res){
  res %>%
    label_sig_enrichments %>%
    ggplot(aes(x=log10(oddsRatio), y=-log10(pFishersExact), color=result)) +
    geom_point() +
    geom_point(
      res %>% dplyr::filter(in_cs, active_cs),
      mapping=aes(x=log10(oddsRatio), y=-log10(pFishersExact)),
      color='black', pch=21, size=5) +
    scale_color_manual(values = c('depleted' = 'coral',
                                  'enriched' = 'dodgerblue',
                                  'not significant' = 'grey'))
}


# take credible set summary, return "best" row for each gene set
get_cs_summary_condensed = function(fit){
  fit %>%
    get_credible_set_summary() %>%
    group_by(geneSet) %>%
    arrange(desc(alpha)) %>%
    dplyr::filter(row_number() == 1)
}
# generate table for making gene-set plots
get_plot_tbl = function(fits, ora){
  res <- fits %>%
    left_join(ora) %>%
    dplyr::mutate(
      gs_summary = map(fit, get_gene_set_summary),
      cs_summary = map(fit, get_cs_summary_condensed),
      res = map2(gs_summary, cs_summary, ~ left_join(.x, .y, by='geneSet')),
      res = map2(res, ora, ~ possibly(left_join, NULL)(.x, .y))
    ) %>%
    dplyr::select(-c(fit, susie.args, ora, gs_summary, cs_summary)) %>%
    unnest(res)
  return(res)
}

# split tibble into a list using 'col'
split_tibble = function(tibble, col = 'col'){
  tibble %>% split(., .[, col])
}

# Get summary of credible sets with gene set descriptions
get_table_tbl = function(fits, ora){
  res2 <- fits %>%
    left_join(ora) %>%
    dplyr::mutate(res = map(fit, get_credible_set_summary)) %>%
    dplyr::mutate(res = map2(res, ora, ~ left_join(.x, .y))) %>%
    dplyr::select(-c(fit, ora)) %>%
    unnest(res)

  descriptions <- map_dfr(genesets, ~pluck(.x, 'geneSet', 'geneSetDes'))
  tbl <- res2 %>%
    dplyr::filter(active_cs) %>%
    left_join(descriptions)
  tbl_split <- split_tibble(tbl, 'experiment')
  html_tables <- map(tbl_split, ~split_tibble(.x, 'db'))
  return(html_tables)
}


#' Report credible set based summary of SuSiE
#' @param tbl output of \alias{get_credible_set_summary} to be formatted
#' @param target_coverage the coverage of the credible sets to be reported
#' @param max_coverage report SNPs that are not in the target_coverage c.s. up this value
#' @param max_sets maximum number of gene sets to report for a single credible set
#'    this is useful for large credible sets
#' @return A styled kable table suitable for rendering in an HTML document
#' @export
report_susie_credible_sets = function(tbl,
                                      target_coverage=0.95,
                                      max_coverage=0.99,
                                      max_sets=10){
  require(kableExtra)
  tbl_filtered <-
    tbl %>%
    group_by(component) %>%
    arrange(component, desc(alpha)) %>%
    dplyr::filter(cumalpha <= max_coverage, alpha_rank <= max_sets) %>%
    dplyr::mutate(in_cs = cumalpha <= target_coverage) %>% 
    dplyr::ungroup() %>%
    dplyr::mutate(logOddsRatio = log10(oddsRatio))

  tbl_filtered %>%
    dplyr::select(component, geneSet, description, geneSetSize, overlap, logOddsRatio, beta, beta.se, alpha, pip, pFishersExact) %>%
    dplyr::mutate_if(is.numeric, funs(as.character(signif(., 3)))) %>%
    pack_group %>%
    column_spec(c(4), color=ifelse(tbl_filtered$beta > 0, 'green', 'red')) %>%
    kableExtra::kable_styling()
}
