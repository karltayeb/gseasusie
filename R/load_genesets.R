make.gene.set <- function(X){
  referenceGene <- rownames(X)
  geneSets <- colnames(X)
  a <- which(X!=0, arr.ind = T)
  geneSet <- cbind(referenceGene[a[,1]], geneSets[a[,2]])
  colnames(geneSet) <- c('gene', 'geneSet')
  geneSet <- as_tibble(geneSet)
  return(geneSet)
}

make.gene.set.matrix <- function(geneSet){
  X <- geneSet %>%
    dplyr::mutate(in_geneSet = 1) %>%
    dplyr::select(geneSet, in_geneSet, gene) %>%
    tidyr::pivot_wider(names_from = geneSet, values_from = in_geneSet) %>%
    dplyr::mutate_at(dplyr::vars(-gene), ~tidyr::replace_na(.x, 0))

  backgroundGene <- X$gene
  X_mat <- as.matrix(X[,-1])
  rownames(X_mat) <- X$gene
  return(Matrix::Matrix(X_mat, sparse = T))
}

load.msigdb.X <- function(){
  msigdb.tb <- msigdbr::msigdbr(species="Homo sapiens", category = c("C2"))

  msigdb.geneSet <- msigdb.tb %>%
    dplyr::select(gs_id, human_entrez_gene) %>%
    dplyr::transmute(geneSet = gs_id, gene = human_entrez_gene)

  X <- make.gene.set.matrix(msigdb.geneSet)
  X <- X[, sample(colnames(X), 500)]
  X <- X[Matrix::rowSums(X) > 0, ]
}

load.webGestalt.geneSet <- function(db='geneontology_Biological_Process_noRedundant'){
  organism <- 'hsapiens'
  interestGeneType = "ensembl_gene_id"
  referenceGeneType = "ensembl_gene_id"
  outputDirectory = './data/WebGestalt/results'
  hostName = 'http://www.webgestalt.org/'

  enrichDatabase <- c(db)
  geneSet <- WebGestaltR::loadGeneSet(
    organism=organism, enrichDatabase=enrichDatabase, hostName=hostName)
  return(geneSet)
}

load.webGestalt.X <- function(db='geneontology_Biological_Process_noRedundant'){
  organism <- 'hsapiens'
  interestGeneType = "ensembl_gene_id"
  referenceGeneType = "ensembl_gene_id"
  outputDirectory = './data/WebGestalt/results'
  hostName = 'http://www.webgestalt.org/'

  enrichDatabase <- c(db)
  geneSet <- WebGestaltR::loadGeneSet(
    organism=organism, enrichDatabase=enrichDatabase, hostName=hostName)$geneSet

  X <- make.gene.set.matrix(geneSet)
  return(X)
}

load.gonr.geneSet <- function(){
  load.webGestalt.geneSet()
}

load.gobp.geneSet <- function(){
  load.webGestalt.geneSet()
}

load.gobp.X <- function(){
  X <- load.webGestalt.X(db='geneontology_Biological_Process')
  sizes <- Matrix::colSums(X)
  X <- X[, (sizes > 5) & (sizes < 500)]
}

load.gonr.X <- function(){
  X <- load.webGestalt.X()
  sizes <- Matrix::colSums(X)
  X <- X[, (sizes > 5) & (sizes < 500)]
}

load.gobp.X <- function(){
  X <- load.webGestalt.X(db='geneontology_Biological_Process')
  sizes <- Matrix::colSums(X)
  X <- X[, (sizes > 5) & (sizes < 500)]
}

load_pathways_genesets <- function(){
  X <- pathways::gene_sets_human$gene_sets
  rownames(X) <- pathways::gene_sets_human$gene_info$GeneID
  geneSetDes <- pathways::gene_sets_human$gene_set_info %>%
    dplyr::rename(
      description = name,
      geneSet = id
    )
  geneSet <- list(geneSetDes = geneSetDes)
  res <- list(X=X, geneSet=geneSet, db='pathways', min.size=1)
}


# turn a tibble with two columns
# into a named list with names from first column
# values from second column
tibble2namedlist <- function(tibble){
  x <- tibble[[2]]
  names(x) <- tibble[[1]]
  return(x)
}

# turn webgestalt geneSet object
# into named list of gene sets
convertGeneSet <- function(gs, min.size = 100){
  gs$geneSet %>%
    dplyr::group_by(geneSet) %>%
    dplyr::filter(dplyr::n() > min.size) %>%
    dplyr::select(gene) %>%
    tidyr::chop(gene) %>% dplyr::ungroup() %>%
    tibble2namedlist
}

# convert list of gene sets into binary indicator matrix
geneSet2X <- function(gs){
  unique.genes <- unique(unlist(gs))
  X <- purrr::map(gs, ~ Matrix::Matrix(unique.genes %in% .x %>% as.integer, sparse = T)) %>%
    Reduce(cbind2, .) %>%
    `rownames<-`(unique.genes) %>%
    `colnames<-`(names(gs))
  return(X)
}

load_webgestalt_geneset_x = function(db, min.size=50){
  res <- xfun::cache_rds({
      gs <- load.webGestalt.geneSet(db)
      X <- gs %>% convertGeneSet(min.size=min.size) %>% geneSet2X
      list(geneSet = gs, X=X, db=db, min.size=min.size)
    }, dir='cache/resources/', file=paste0(db, '.', min.size, '.X.rds')
  )
  return(res)
}

#' load MSigDB gene sets from `msigdbr` package
#' @param db is the MSigDB category to fetch (C1, C2, ... C6)
#' @param min.size remove gene sets smaller than this (default 10)
#' @export
load_msigdb_geneset_x <- function(db='C2', min.size=10){
  res <- xfun::cache_rds({
    msigdb.tb <- msigdbr::msigdbr(species="Homo sapiens", category = db)
    geneSetDes <- msigdb.tb %>% 
      dplyr::select(gs_id, gs_cat, gs_subcat, gs_description)
    gs <- msigdb.tb %>%
      dplyr::select(gs_id, human_entrez_gene) %>%
      dplyr::transmute(geneSet = gs_id, gene = human_entrez_gene) %>%
      dplyr::distinct() %>%
      list(geneSet = ., geneSetDes = geneSetDes)
    X <- gs %>% convertGeneSet(min.size= min.size) %>% geneSet2X
    list(geneSet = gs, X=X, db=db, min.size=min.size)
  }, dir='cache/resources/', file=paste0(db, '.', min.size, '.X.rds'))
  return(res)
}

#' load gene sets from various sources with uniform format
#' @export
load_gene_sets = function(dbs=c('gobp', 'gobp_nr', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'pathways')) {
  load_pathways_genesets <- load_pathways_genesets
  promise <- tibble::tribble(
    ~name, ~expression,
    'gobp', rlang::expr(gseareg:::load_webgestalt_geneset_x('geneontology_Biological_Process', min.size=50)),
    'gobp_nr', rlang::expr(gseareg:::load_webgestalt_geneset_x('geneontology_Biological_Process_noRedundant', min.size=1)),
    'gomf', rlang::expr(gseareg:::load_webgestalt_geneset_x('geneontology_Molecular_Function', min.size=1)),
    'gocc', rlang::expr(gseareg:::load_webgestalt_geneset_x('geneontology_Cellular_Component', min.size=1)),
    'kegg', rlang::expr(gseareg:::load_webgestalt_geneset_x('pathway_KEGG', min.size=1)),
    'c1', rlang::expr(gseareg:::load_msigdb_geneset_x('C1', min.size=1)),
    'c2', rlang::expr(gseareg:::load_msigdb_geneset_x('C2', min.size=1)),
    'c3', rlang::expr(gseareg:::load_msigdb_geneset_x('C3', min.size=1)),
    'c4', rlang::expr(gseareg:::load_msigdb_geneset_x('C4', min.size=1)),
    'c5', rlang::expr(gseareg:::load_msigdb_geneset_x('C5', min.size=1)),
    'c6', rlang::expr(gseareg:::load_msigdb_geneset_x('C6', min.size=1)),
    'pathways', rlang::expr(gseareg:::load_pathways_genesets())
  )
  genesets <-
    promise %>%
    dplyr::filter(name %in% dbs) %>%
    dplyr::mutate(geneset = purrr::map(expression, eval)) %>%
    dplyr::select(name, geneset) %>%
    tibble2namedlist()
  return(genesets)
}

#' take a list of genesets and concatenate them
#' @export
concat_genesets = function(genesets){
  # intersect/union of genes
  all_genes <- map(genesets, ~ .x$X %>% rownames())
  common_genes <- Reduce(intersect, all_genes)
  union_genes <- unique(unlist(all_genes))

  # subset and order (how to make this work for union?)
  X_sub <- map(genesets, ~with(.x, X[rownames(X) %in% common_genes, ] %>% {.[order(rownames(.)),]}))

  return(genesets)
}
