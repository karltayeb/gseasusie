#' convert geneSet tibble to matrix
geneSet2X <- function(gs){
  unique.genes <- unique(unlist(gs))
  X <- purrr::map(gs, ~ Matrix::Matrix(unique.genes %in% .x %>% as.integer, sparse = T)) %>%
    Reduce(cbind2, .) %>%
    `rownames<-`(unique.genes) %>%
    `colnames<-`(names(gs))
  return(X)
}

#' convert gene set matrix to geneSet tibble
X2geneSet = function(X){
  referenceGene <- rownames(X)
  geneSets <- colnames(X)
  a <- BiocGenerics::which(X!=0, arr.ind = T)
  geneSet <- cbind(referenceGene[a[,1]], geneSets[a[,2]])
  colnames(geneSet) <- c('gene', 'geneSet')
  geneSet <- tibble::as_tibble(geneSet)
  return(geneSet)
}

# turn a tibble with two columns
# into a named list with names from first column
# values from second column
tibble2namedlist <- function(tibble){
  x <- tibble[[2]]
  names(x) <- tibble[[1]]
  return(x)
}

#' convert a geneSet tibble into a list of gene sets
#'
#' @param gs a tibble with two at least two columns: geneSet and gene
#' @param min.size the minimum size to retain a gene set
#' returns a named list mapping gene sets to genes
convertGeneSet <- function(geneSet, min.size = 100){
  geneSet %>%
    dplyr::group_by(geneSet) %>%
    dplyr::filter(dplyr::n() >= min.size) %>%
    dplyr::select(gene) %>%
    tidyr::chop(gene) %>% dplyr::ungroup() %>%
    tibble2namedlist
}


#' wrapper for loading geneSets from peters `pathways` package
load_pathways_genesets <- function(){
  X <- pathways::gene_sets_human$gene_sets
  rownames(X) <- pathways::gene_sets_human$gene_info$GeneID
  geneSetDes <- pathways::gene_sets_human$gene_set_info %>%
    dplyr::rename(
      description = name,
      geneSet = id
    )

  geneSet <- X2geneSet(X)
  geneSet <- list(geneSetDes = geneSetDes)
  res <- list(X=X, geneSet=geneSet, db='pathways', min.size=1)
  res <- list(X=X, geneSet=geneSet, geneSetDes=geneSetDes, db='pathways', min.size=1)
}

#' wrapper for webGestatlR geneSet loading
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

#' format WebGestaltR geneSets to standardized format
load_webgestalt_geneset_x = function(db, min.size=10){
  message(paste0('loading gene set from webgestaltr: ', db))
  res <- xfun::cache_rds({
      gs <- load.webGestalt.geneSet(db)
      X <- gs$geneSet %>% convertGeneSet(min.size=min.size) %>% geneSet2X
      list(X=X, geneSet=gs$geneSet, geneSetDes=gs$geneSetDes, db=db, min.size=min.size)
    }, dir='cache/resources/', file=paste0(db, '.', min.size, '.X.rds')
  )
  return(res)
}

#' load MSigDB gene sets from `msigdbr` package in standardized format
#' @param db is the MSigDB category to fetch (C1, C2, ... C6)
#' @param min.size remove gene sets smaller than this (default 10)
#' @export
load_msigdb_geneset_x <- function(db='C2', min.size=10){
  message(paste0('loading gene set from msigdbr: ', db))
  res <- xfun::cache_rds({
    msigdb.tb <- msigdbr::msigdbr(species="Homo sapiens", category = db)
    geneSetDes <- msigdb.tb %>%
      dplyr::select(gs_id, gs_cat, gs_subcat, gs_description) %>%
      dplyr::distinct() %>%
      dplyr::rename(geneSet=gs_id, description=gs_description)
    gs <- msigdb.tb %>%
      dplyr::select(gs_id, human_entrez_gene) %>%
      dplyr::transmute(geneSet = gs_id, gene = human_entrez_gene) %>%
      dplyr::mutate(gene = as.character(gene)) %>%
      dplyr::distinct() %>%
      list(geneSet = ., geneSetDes = geneSetDes)
    X <- gs$geneSet %>% convertGeneSet(min.size= min.size) %>% geneSet2X
    list(X=X, geneSet=gs$geneSet, geneSetDes=gs$geneSetDes, db=db, min.size=min.size)
  }, dir='cache/resources/', file=paste0(db, '.', min.size, '.X.rds'))
  return(res)
}

#' take a list of genesets and concatenate them
#' @param genesets a list of gene sets like the output from load_gene_sets
#' @param min.size minimum size of gene set for inclusion
#' @param name a name for this geneset (it get's cached for fast loading in the future)
#' @export
concat_genesets = function(genesets, min.size, name){
  res <- xfun::cache_rds({
    geneSet <- purrr::map_dfr(genesets, ~purrr::pluck(.x, 'geneSet')) %>%
      dplyr::select(geneSet, gene) %>%
      dplyr::distinct()
    geneSetDes <- purrr::map_dfr(genesets, ~purrr::pluck(.x, 'geneSetDes')) %>%
      dplyr::select(geneSet, description) %>%
      dplyr::distinct()
    X <- geneSet %>% convertGeneSet(min.size= min.size) %>% geneSet2X
    list(X=X, geneSet=geneSet, geneSetDes=geneSetDes, db=name, min.size=min.size)
  }, dir='cache/resources/', file=paste0(name, '.', min.size, '.X.rds'))
  return(res)
}

#' convenient function to load MSigDb C1-6
#' @param min.size minimum number of genes in term
#' @export
load_all_msigdb = function(min.size=10){
  genesets <- load_gene_sets(c('c1', 'c2', 'c3', 'c4', 'c5', 'c6'))
  gs <- concat_genesets(genesets, min.size, 'all_msigdb')
  return(gs)
}

#' convenient function to load all GO terms (across ontologies BP, MF, CC)
#' @param min.size minimum number of genes in term (default 10)
#' @export
load_all_go = function(min.size=10){
  genesets <- gseasusie::load_gene_sets(c('gobp', 'gomf', 'gocc'))
  go.gs <- concat_genesets(genesets, min.size, 'all_go')
  return(go.gs)
}

#' load gene sets from various sources with uniform format
#' @export
load_gene_sets = function(dbs=c('gobp', 'gobp_nr', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'pathways')) {
  promise <- tibble::tribble(
    ~name, ~expression,
    'gobp_nr', rlang::expr(gseasusie:::load_webgestalt_geneset_x('geneontology_Biological_Process_noRedundant', min.size=1)),
    'gobp', rlang::expr(gseasusie:::load_webgestalt_geneset_x('geneontology_Biological_Process', min.size=1)),
    'gomf', rlang::expr(gseasusie:::load_webgestalt_geneset_x('geneontology_Molecular_Function', min.size=1)),
    'gocc', rlang::expr(gseasusie:::load_webgestalt_geneset_x('geneontology_Cellular_Component', min.size=1)),
    'kegg', rlang::expr(gseasusie:::load_webgestalt_geneset_x('pathway_KEGG', min.size=1)),
    'c1', rlang::expr(gseasusie:::load_msigdb_geneset_x('C1', min.size=1)),
    'c2', rlang::expr(gseasusie:::load_msigdb_geneset_x('C2', min.size=1)),
    'c3', rlang::expr(gseasusie:::load_msigdb_geneset_x('C3', min.size=1)),
    'c4', rlang::expr(gseasusie:::load_msigdb_geneset_x('C4', min.size=1)),
    'c5', rlang::expr(gseasusie:::load_msigdb_geneset_x('C5', min.size=1)),
    'c6', rlang::expr(gseasusie:::load_msigdb_geneset_x('C6', min.size=1)),
    'c7', rlang::expr(gseasusie:::load_msigdb_geneset_x('C7', min.size=1)),
    'c8', rlang::expr(gseasusie:::load_msigdb_geneset_x('C8', min.size=1)),
    'h', rlang::expr(gseasusie:::load_msigdb_geneset_x('H', min.size=1)),
    'pathways', rlang::expr(gseasusie:::load_pathways_genesets()),
    'all_msigdb', rlang::expr(gseasusie:::load_all_msigdb()),
    'all_go', rlang:::expr(gseasusie:::load_all_go())
  )
  genesets <-
    promise %>%
    dplyr::filter(name %in% dbs) %>%
    dplyr::mutate(geneset = purrr::map(expression, eval)) %>%
    dplyr::select(name, geneset) %>%
    tibble2namedlist()
  return(genesets)
}
