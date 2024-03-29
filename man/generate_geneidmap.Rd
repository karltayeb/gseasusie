% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gseasusie.R
\name{generate_geneidmap}
\alias{generate_geneidmap}
\title{Make a tibble with two columns for mapping between two gene IDs
Optionally, filter down to unique gene id's in both columns}
\usage{
generate_geneidmap(genes, from, to, filter = TRUE)
}
\arguments{
\item{genes}{a list of genes of}

\item{from}{the gene ID type of \code{genes} must be in \code{AnnotationDbi::columns(org.Hs.eg.db::org.Hs.eg.db)}}

\item{to}{the gene ID type we are mapping to, must be in \code{AnnotationDbi::columns(org.Hs.eg.db::org.Hs.eg.db)}}
}
\description{
Make a tibble with two columns for mapping between two gene IDs
Optionally, filter down to unique gene id's in both columns
}
