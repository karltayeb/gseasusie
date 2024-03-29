% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/load_genesets.R
\name{convertGeneSet}
\alias{convertGeneSet}
\title{convert a geneSet tibble into a list of gene sets}
\usage{
convertGeneSet(geneSet, min.size = 100)
}
\arguments{
\item{min.size}{the minimum size to retain a gene set
returns a named list mapping gene sets to genes}

\item{gs}{a tibble with two at least two columns: geneSet and gene}
}
\description{
convert a geneSet tibble into a list of gene sets
}
