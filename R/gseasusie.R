#' Subset gene set matrix to genes present in test_background, and discard gene sets with too few genes
subset_gene_set_matrix <- function(X, test_background, min_genes=5){
  X_sub <- X[test_background,]
  genes_per_gs <- BiocGenerics::colSums(X_sub)
  X_sub <- X_sub[, genes_per_gs > min_genes]
}

#' Make a tibble with two columns for mapping between two gene IDs
#' Optionally, filter down to unique gene id's in both columns
#' @param genes a list of genes of
#' @param from the gene ID type of `genes` must be in `AnnotationDbi::columns(org.Hs.eg.db::org.Hs.eg.db)`
#' @param to the gene ID type we are mapping to, must be in `AnnotationDbi::columns(org.Hs.eg.db::org.Hs.eg.db)`
generate_geneidmap = function(genes, from, to, filter=TRUE){
  hs <- org.Hs.eg.db::org.Hs.eg.db
  genes <- unique(genes)

  if(from == to){  # case: no need to convert
    idmap <- tibble::tibble(genes)
    colnames(idmap) <- from
  } else{
    idmap <- AnnotationDbi::select(
      hs, keys=genes,
      columns=c(to, from),
      keytype = from)
  }


  # filter to unique values THIS MAY MISS A FEW GENES
  if(filter){
    idmap <- idmap %>%
      dplyr::distinct(!!rlang::sym(from), .keep_all = T) %>%
      dplyr::distinct(!!rlang::sym(to), .keep_all = T)
  }
  return(idmap)
}

#' Covert Labels and restrict to test background
#' @param study_background a list of genes (ID type `from`)
#' @param background a list of genes (ID type `to`)
#' @param study_id_type the gene ID type of `study_background` must be in `AnnotationDbi::columns(org.Hs.eg.db::org.Hs.eg.db)`
#' @param background_id_type the gene ID type of `background` must be in `AnnotationDbi::columns(org.Hs.eg.db::org.Hs.eg.db)`
generate_test_background_map <- function(study_background,
                                         background,
                                         study_id_type='SYMBOL',
                                         background_id_type='ENTREZID'){

  # generate id map
  g2g <- generate_geneidmap(background,
                            from=background_id_type,
                            to=study_id_type)

  # set study background to background if not specified
  if(is.null(study_background)){
    warning('No study background specified, using all genes in db...')
    study_background <- g2g[[study_id_type]]
  }

  # filter to genes in study background AND background. This is test_background
  g2g <- g2g %>% dplyr::filter(
    !!rlang::sym(study_id_type) %in% study_background,
    !!rlang::sym(background_id_type) %in% background)
  return(g2g)
}

#' Gene List to Binary Vector
#' Convert a list of genes to a binary vector against a set of background genes
#' This function assumes the gene list is a strict subset of the background
#' @param gene_list a list of genes
#' @param background a list of background genes, length n
#' @return a binary vector of length n indicating if the background gene is present in the gene list
convert_genelist2binaryvector <- function(gene_list, background){
  stopifnot("`gene_list` is not a subset of `background`"=length(setdiff(gene_list, background)) == 0)
  genes <- as.integer(background %in% gene_list)
  names(genes) <- background
  return(genes)
}

#' Prep gene list
#' Converts a list of genes to input for logistic SuSiE.
#' It is important that enrichment be assessed on an appropriate background,
#' so this function subsets to the intersection of genes in the study background and the union of genes in any gene set
#' @param gene_list a list of genes
#' @param study_background the background on which `gene_list` was selected
#' @param study_id_type the gene ID type of gene_list, from `AnnotationDbi::columns(org.Hs.eg.db::org.Hs.eg.db)`
#' @param db the name of the pathway database(s) to use. Must be in `gseasusie::list_db`
#' @return a list with elements X and y suitable for binary fit functions (fit_logistic_susie, fit_ora, fit_marginal_regression, etc)
prep_list_data <- function(gene_list, study_background=NULL, study_id_type='SYMBOL', db='pathways'){
  # load db and extract background genes
  DB <- gseasusie::load_gene_sets(db)
  background <- rownames(DB[[db]]$X)
  background_id_type = 'ENTREZID'  # TODO: better to have this info stored in DB, rather than enforcing standard

  # get test_background map
  g2g <- generate_test_background_map(study_background, background, study_id_type, background_id_type)

  # map ids in gene list
  gene_list_converted <- g2g %>%
    dplyr::filter(!!rlang::sym(study_id_type) %in% gene_list) %>%
    dplyr::pull(!!rlang::sym(background_id_type))

  test_background <- g2g %>% dplyr::pull(!!rlang::sym(background_id_type))

  # genes that did not make it into test_background
  gene_list_excluded <- setdiff(gene_list, g2g %>% dplyr::pull(!!rlang::sym(study_id_type)))
  study_background_excluded <- setdiff(study_background, g2g %>% dplyr::pull(!!rlang::sym(study_id_type)))
  background_excluded <- setdiff(background, g2g %>% dplyr::pull(!!rlang::sym(background_id_type)))

  # convert to binary vector
  y <- convert_genelist2binaryvector(gene_list_converted, test_background)

  # subset gene set matirx
  X <- subset_gene_set_matrix(DB$pathways$X, test_background)
  X.sets <- colnames(X)

  data <- list(
    y = y,
    X = X,
    test_background = test_background,
    study_background = study_background,
    background = background,
    gene_list_excluded = gene_list_excluded,
    study_background_excluded = study_background_excluded,
    background_excluded = background_excluded,
    X.sets = colnames(X)
  )
}

#' takes gene_list, study_backround, and geneset
prep_list_data2 <- function(gene_list, study_background=NULL, study_id_type='SYMBOL', geneset){
  # load db and extract background genes
  background <- rownames(geneset$X)
  background_id_type = 'ENTREZID'  # TODO: better to have this info stored in DB, rather than enforcing standard

  # get test_background map
  g2g <- generate_test_background_map(study_background, background, study_id_type, background_id_type)

  # map ids in gene list
  gene_list_converted <- g2g %>%
    dplyr::filter(!!rlang::sym(study_id_type) %in% gene_list) %>%
    dplyr::pull(!!rlang::sym(background_id_type))

  test_background <- g2g %>% dplyr::pull(!!rlang::sym(background_id_type))

  # genes that did not make it into test_background
  gene_list_excluded <- setdiff(gene_list, g2g %>% dplyr::pull(!!rlang::sym(study_id_type)))
  study_background_excluded <- setdiff(study_background, g2g %>% dplyr::pull(!!rlang::sym(study_id_type)))
  background_excluded <- setdiff(background, g2g %>% dplyr::pull(!!rlang::sym(background_id_type)))

  # convert to binary vector
  y <- convert_genelist2binaryvector(gene_list_converted, test_background)

  # subset gene set matirx
  X <- subset_gene_set_matrix(geneset$X, test_background)
  X.sets <- colnames(X)

  data <- list(
    y = y,
    X = X,
    test_background = test_background,
    study_background = study_background,
    background = background,
    gene_list_excluded = gene_list_excluded,
    study_background_excluded = study_background_excluded,
    background_excluded = background_excluded,
    X.sets = colnames(X)
  )
}


list_driver <- function(gene_list, study_background=NULL, study_id_type='SYMBOL', db='pathways'){
  # 1.  prepare data
  data <- prep_list_data(gene_list,
                         study_background = study_background,
                         study_id_type = study_id_type,
                         db = db)

  # 2. fit
  #logsusie_fit <- fit_logistic_susie_veb_boost(data$X, data$y)
  susie_fit <- susieR::susie(data$X, data$y)
  ora <- fit_ora(data$X, data$y)

  # 3. post process?
  res <- list(
    ora=ora,
    logsusie_fit = logsusie_fit,
    susie_fit=susie_fit,
    data=data
  )
  return(list(ora=ora, logsusie_fit = logsusie_fit, susie_fit=susie_fit, data=data))
}


#' Score Driver
#' Run GSEA on gene level statistics
prep_score_data <- function(gene_scores, gene_scores_sd=NULL, study_id_type='SYMBOL', db='pathways'){
  # load db and extract background genes
  DB <- gseasusie::load_gene_sets(db)
  background <- rownames(DB[[db]]$X)
  background_id_type = 'ENTREZID'  # TODO: better to have this info stored in DB, rather than enforcing standard

  # default SDs
  if(is.null(gene_scores_sd)){
    warning('Gene score standard errors not provided, defaulting to 1 (assume gene scores are z-scores)')
    gene_scores_sd <- rep(1, length(gene_scores))
    names(gene_scores_sd) <- names(gene_scores)
  }

  # get test_background map
  study_background <- names(gene_scores)
  g2g <- generate_test_background_map(study_background, background, study_id_type, background_id_type)
  test_background <- g2g[[background_id_type]]

  # subset scores and sds
  gene_scores_test <- gene_scores[g2g[[study_id_type]]]
  names(gene_scores_test) <- g2g[[background_id_type]]

  gene_scores_sd_test <- gene_scores_sd[g2g[[study_id_type]]]
  names(gene_scores_sd_test) <- g2g[[background_id_type]]

  # subset gene set matrix
  X <- subset_gene_set_matrix(DB$pathways$X, test_background)

  # genes that did not make it into test_background
  gene_list_excluded <- setdiff(gene_list, g2g[[study_id_type]])
  study_background_excluded <- setdiff(study_background, g2g[[study_id_type]])
  background_excluded <- setdiff(background,  g2g[[background_id_type]])

  data <- list(
    betahat = gene_scores_test,
    betahat_se = gene_scores_sd_test,
    test_background = test_background,
    study_background = study_background,
    background = background,
    gene_list_excluded = gene_list_excluded,
    study_background_excluded = study_background_excluded,
    background_excluded = background_excluded,
    X.sets = colnames(X),
    X = X
  )

  return(data)
}

score_driver <- function(gene_scores, gene_scores_sd=NULL, study_id_type='SYMBOL', db='pathways'){
  data <- prep_score_data(gene_scores,
                          gene_scores_sd=gene_scores_sd,
                          study_id_type=study_id_type,
                          db=db)
  pnsusie_fit <- logisticsusie::pointnormalsusie(as.matrix(data$X), data$betahat, data$betahat_se)
  return(list(
    pnsusie_fit = pnsusie_fit
  ))
}


#data <- prep_score_data(gene_scores, NULL, 'SYMBOL', 'pathways')
#pnsusie <- score_driver(gene_scores)


#' @export
coerce_list <- function(list, background, gs_background, from='ENSEMBL', to='ENTREZID'){
  # map unique IDs
  idmap <- gseasusie:::generate_geneidmap(background, from, to, filter = T)
  colnames(idmap) <- c('from', 'to')

  # subset gene list background to genes that can be mapped
  background2 <- intersect(background, idmap$from)
  # map gene ids
  background3 <- idmap[idmap$from %in% background2,]$to
  # intersect with gene set background
  background4 <- intersect(gs_background, background3)

  # get gene list, restricted to shared background genes
  idmap <- idmap[idmap$to %in% background4,]
  list4 <- idmap[idmap$from %in% list,]$to

  res <- list(list = list4, background = background4)
  return(res)
}

#' Coerce scores
#'
#' @param de a data.frame or similar
#' @export
coerce_scores <- function(de, gs_background, from='ENSEMBL', to = 'ENTREZID'){
  # get gene names from rownames, assign to column ID
  background <- rownames(de)
  de <- de %>% mutate(ID = rownames(de))

  # map unique IDs
  idmap <- gseasusie:::generate_geneidmap(background, from, to, filter = T)
  colnames(idmap) <- c('from', 'to')

  # subset gene list background to genes that can be uniquely mapped
  background2 <- intersect(background, idmap$from)
  # map gene ids
  background3 <- idmap[idmap$from %in% background2,]$to
  # intersect with gene set background
  background4 <- intersect(gs_background, background3)

  # get gene list, restricted to shared background genes
  idmap <- idmap[idmap$to %in% background4,]
  de2 <- idmap %>% inner_join(de, by=join_by(from == ID))
  return(de2)
}
