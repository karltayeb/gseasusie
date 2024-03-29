% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/enrichment_pipeline.R
\name{prep_binary_data}
\alias{prep_binary_data}
\title{Pepare gene set matrix and binary gene list}
\usage{
prep_binary_data(gs, dat, thresh = 0.01, ptop = NULL, .sign = c(1, -1))
}
\arguments{
\item{gs}{a list containing gene set matrix \code{X}
with genes in \code{rownames} and gene set names in `colnames}

\item{dat}{a vector of statistics to threshold hold,
names must match columns of \code{X}}

\item{ptop}{proportion of genes to include in gene list, if not NULL, overrides thresh}

\item{.sign}{either \code{c(1, -1)}, 1, or -1, to indicate including both, positive, or negative betas}

\item{threshold}{to binarize \code{dat}}
}
\description{
Pepare gene set matrix and binary gene list
}
