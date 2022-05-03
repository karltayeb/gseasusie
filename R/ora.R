compute_fet_pval = function(overlap, geneListSize, geneSetSize, nGenes){
ct <- matrix(c(
    overlap,
    geneListSize-overlap,
    geneSetSize-overlap,
    nGenes - geneSetSize - geneListSize + overlap), nr=2)
return(fisher.test(ct)$p.value)
}

compute_hypergeometric_pval = function(overlap, geneListSize, geneSetSize, nGenes){
# genes in list = white balls, not in list = black balls
# draw k balls w/o replacement where k is the size of the gene set
# is the overlap we see with our gene set extreme/unlikely?
return(phyper(
    max(0, overlap - 1),  # p(X >= overlap) = p(X > overlap - 1)
    geneListSize,
    nGenes - geneListSize,
    geneSetSize,
    lower.tail = FALSE))
}

#' @export
fit_ora = function(X, y){
  ora <- tibble(
      geneSet = colnames(X),
      geneListSize = sum(y),
      geneSetSize = BiocGenerics::colSums(X),
      overlap = (y %*% X)[1,],
      nGenes = length(y),
      propInList = overlap / geneListSize,
      propInSet = overlap / geneSetSize,
      oddsRatio = (overlap * (nGenes - geneSetSize - geneListSize + overlap)) /
        ((geneListSize-overlap) * (geneSetSize - overlap))
    ) %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      pHypergeometric = compute_hypergeometric_pval(overlap, geneListSize, geneSetSize, nGenes),
      pFishersExact = compute_fet_pval(overlap, geneListSize, geneSetSize, nGenes)
    ) %>%
    dplyr::ungroup()
    return(ora)
}
