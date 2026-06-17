if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("org.Mm.eg.db")

library(org.Mm.eg.db)
library(AnnotationDbi)

genes = readLines("~/Downloads/neha/ensembl_to_ucsc_conversion_2/genes.txt")
res <- select(org.Mm.eg.db,
              keys = genes,
              keytype = 'ENSEMBL',
              columns = c("REFSEQ")
)

head(res)
