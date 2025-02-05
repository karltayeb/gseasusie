logsumexp <- function(x) {
  log(sum(exp(x - max(x)))) + max(x)
}

select_db <- function(organism) {
  switch(tolower(organism),
    hsapiens = org.Hs.eg.db::org.Hs.eg.db,
    mmusculus = org.Mm.eg.db::org.Mm.eg.db,
    stop("Unsupported organism")
  )
}

get_go_info <- function(fit) {
  AnnotationDbi::select(GO.db::GO.db,
    keys = unique(fit$data$geneSets$geneSet),
    columns = c("TERM", "DEFINITION"),
    keytype = "GOID"
  ) %>%
    tibble::as_tibble() %>%
    dplyr::mutate(geneSet = GOID)
}

get_gene_info <- function(fit) {
  db <- select_db(fit$organism)
  AnnotationDbi::select(db,
    keys = unique(fit$data$geneMapping$gene),
    columns = c("SYMBOL", "GENETYPE", "GENENAME"),
    keytype = "ENTREZID"
  ) %>%
    tibble::as_tibble() %>%
    dplyr::mutate(gene = ENTREZID)
}


#' Make table summarizing credible sets
#' @export
make_cs_tbl <- function(fit, coverage = 0.9, min_log2bf_ser = log2(10), max_gs_per_component = as.integer(1e9)) {
  go_info <- get_go_info(fit)
  gene_info <- fit$data$geneSets %>%
    dplyr::left_join(get_gene_info(fit)) %>%
    dplyr::group_by(geneSet) %>%
    dplyr::mutate(size = length(geneSet), propInList = mean(geneInList))

  fit %>%
    make_component_tbl(min_lbf_ser = min(min_log2bf_ser * log(2), max(fit$fit$lbf_ser))) %>%
    dplyr::group_by(component) %>%
    dplyr::arrange(desc(alpha)) %>%
    dplyr::slice_head(n = max_gs_per_component) %>%
    dplyr::mutate(
      cumalpha = cumsum(alpha),
      in_cs = (dplyr::row_number() <= which(cumalpha >= coverage)[1]),
      lbf_ser = logsumexp(lbf) - log(length(lbf))
    ) %>%
    dplyr::ungroup() %>%
    dplyr::arrange(component, desc(alpha)) %>%
    dplyr::filter(in_cs) %>%
    dplyr::left_join(go_info) %>%
    dplyr::left_join(gene_info) %>%
    # give readable names
    dplyr::mutate(
      log2OR = log2(exp(1)) * beta,
      PIP = alpha,
      log2BF = log2(exp(1)) * lbf,
      log2BF_SER = log2(exp(1)) * lbf_ser,
      name = TERM
    ) %>%
    dplyr::select(c(component, log2BF_SER, geneSet, name, DEFINITION, size, propInList, log2OR, PIP, log2BF)) %>%
    dplyr::distinct() %>%
    dplyr::filter(log2BF_SER >= min_log2bf_ser)
}

#' Create Red - Green column style
make_rwg_column_style <- function(max = 2) {
  function(values) {
    pal <- function(x) rgb(colorRamp(c("red", "white", "green"))(x), maxColorValue = 255)
    clipped <- (pmax(-max, pmin(values, max)) + max) / (2 * max)
    color <- pal(clipped)
    list(background = color)
  }
}

#' Sensible default column defintions
make_sensible_colDef <- function(data) {
  # Initialize an empty list to store colDef objects
  col_defs <- list()
  # Iterate over each column in the data
  for (col_name in names(data)) {
    # Get the column data
    col_data <- data[[col_name]]

    # Determine the type of the column
    col_type <- class(col_data)

    # Set default colDef based on the column type
    if ("numeric" %in% col_type) {
      # Numeric columns: Format to 2 decimal places
      col_defs[[col_name]] <- reactable::colDef(format = reactable::colFormat(digits = 2))
    } else if ("integer" %in% col_type) {
      # Integer columns: No decimal places
      col_defs[[col_name]] <- reactable::colDef(format = reactable::colFormat(digits = 0))
    } else if ("character" %in% col_type || "factor" %in% col_type) {
      # Character or factor columns: Truncate long strings
      # col_defs[[col_name]] <- colDef(truncate = TRUE, minWidth = 150)
    } else if ("Date" %in% col_type) {
      # Date columns: Format as a readable date
      col_defs[[col_name]] <- reactable::colDef(format = reactable::colFormat(date = TRUE))
    } else if ("logical" %in% col_type) {
      # Logical columns: Display as checkboxes
      col_defs[[col_name]] <- reactable::colDef(cell = function(value) {
        if (isTRUE(value)) "✔" else "✗"
      })
    } else {
      # Default colDef for other types
      col_defs[[col_name]] <- reactable::colDef()
    }
  }
  return(col_defs)
}

# this styling helps visually group rows belonging to the same CS
componentColDef <- reactable::colDef(
  style = reactable::JS("function(rowInfo, column, state) {
        const firstSorted = state.sorted[0]
        // Merge cells if unsorted or sorting by school
        if (!firstSorted || firstSorted.id === 'component') {
          const prevRow = state.pageRows[rowInfo.viewIndex - 1]
          if (prevRow && rowInfo.values['component'] === prevRow['component']) {
            return { visibility: 'hidden' }
          }
        }
      }")
)

# this styling displays the log2BF for the SER at the top level
componentlog2BFColDef <- reactable::colDef(
  style = reactable::JS("function(rowInfo, column, state) {
        const firstSorted = state.sorted[0]
        // Merge cells if unsorted or sorting by school
        if (!firstSorted || firstSorted.id === 'log2BF_SER') {
          const prevRow = state.pageRows[rowInfo.viewIndex - 1]
          if (prevRow && rowInfo.values['log2BF_SER'] === prevRow['log2BF_SER']) {
            return { visibility: 'hidden' }
          }
        }
      }"),
  format = reactable::colFormat(digits = 2),
  aggregate = "min"
)

# See the ?tippy documentation to learn how to customize tooltips
with_tooltip <- function(value, tooltip, ...) {
  htmltools::div(
    style = "text-decoration: underline; text-decoration-style: dotted; cursor: help",
    tippy::tippy(value, tooltip, ...)
  )
}

#' Render cs table with reactable
#' @export
make_reactable <- function(cs_table) {
  if (nrow(cs_table) == 0) {
    return(NULL)
  }
  column_format <- make_sensible_colDef(cs_table)
  column_format$geneSet <- reactable::colDef()
  column_format$log2OR <- reactable::colDef(
    style = make_rwg_column_style(2),
    format = reactable::colFormat(
      digits = 2
    )
  )
  column_format$component <- componentColDef
  column_format$log2BF_SER <- componentlog2BFColDef
  term2def <- purrr::map(cs_table$DEFINITION, ~ dplyr::if_else(is.na(.x), "No definition found", .x))
  names(term2def) <- cs_table$geneSet
  column_format$geneSet <- reactable::colDef(cell = function(value) with_tooltip(value, term2def[[value]]))

  reactable::reactable(
    dplyr::select(cs_table, -c(DEFINITION)),
    groupBy = "component",
    columns = column_format,
    paginateSubRows = TRUE,
    defaultExpanded = TRUE
  )
}

#' @export
make_gt_table <- function(cs_table) {
  cs_table %>%
    dplyr::group_by(component) %>%
    dplyr::select(-c(log2BF_SER, DEFINITION)) %>%
    gt::gt() %>%
    gt::fmt_number(columns = c(propInList, log2OR, PIP, log2BF), decimals = 2)
}
