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

loadGeneSetX = function(db, min.size=50){
  res <- xfun::cache_rds({
      gs <- load.webGestalt.geneSet(db)
      X <- gs %>% convertGeneSet(min.size=min.size) %>% geneSet2X
      list(geneSet = gs, X=X, db=db, min.size=min.size)
    }, dir='cache/resources/', file=paste0(db, '.', min.size, '.X.rds')
  )
  return(res)
}

#' load a bunch of genesets
#' @export
load_gene_sets = function() {
  # load genesets
  gobp <- loadGeneSetX('geneontology_Biological_Process', min.size=50)  # just huge number of gene sets
  gobp_nr <- loadGeneSetX('geneontology_Biological_Process_noRedundant', min.size=1)
  gomf <- loadGeneSetX('geneontology_Molecular_Function', min.size=1)
  gocc <- loadGeneSetX('geneontology_Cellular_Component', min.size=1)
  kegg <- loadGeneSetX('pathway_KEGG', min.size=1)
  reactome <- loadGeneSetX('pathway_Reactome', min.size=1)
  wikipathway_cancer <- loadGeneSetX('pathway_Wikipathway_cancer', min.size=1)
  wikipathway <- loadGeneSetX('pathway_Wikipathway', min.size=1)

  genesets <- list(
    gobp=gobp,
    gobp_nr=gobp_nr,
    gomf=gomf,
    gocc=gocc,
    kegg=kegg,
    reactome=reactome,
    wikipathway_cancer=wikipathway_cancer,
    wikipathway=wikipathway
  )

  return(genesets)
}
