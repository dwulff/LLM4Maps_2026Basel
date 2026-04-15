packages = c("tidyverse", "reticulate", "umap", "fastcluster", "wordcloud", 
             "glue", "RSpectra", "jsonlite", "httr", "viridis")
for (p in packages) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p)
    }
  library(p, character.only = TRUE)
  }

if(!py_module_available("sentence_transformers")) py_install("sentence-transformers", pip = TRUE)

# helpers

cosine = function(m) {
  norms = sqrt(rowSums(m^2))
  cos = tcrossprod(m / norms)
  cos[is.na(cos)] = 0
  cos
}

normalize = function(m){
  norms = sqrt(rowSums(m^2))
  norms[norms == 0] = 1
  m = m / norms
  m
}

ppmi = function(m) {
  colsums = colSums(m)**.3
  expected = outer(rowSums(m), colsums) /  sum(colsums)
  pmi = log2(m / expected)
  pmi[pmi < 0 | !is.finite(pmi)] <- 0
  pmi
}

# get qwen

st = import("sentence_transformers")
qwen = st$SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')

# get qwen and setup gemma


text_gen = function(prompt, len = 50L, model = "gemma4:e2b", think = FALSE, verbose = FALSE){
  
  res = POST(
    "http://localhost:11434/api/chat",
    body = list(
      model = "gemma4:e2b",
      messages = list(
        list(role = "user", content = prompt)
      ),
      think = think,
      stream = FALSE,
      options = list(
        num_predict = len   
      )
    ),
    encode = "json"
  )
  
  txt = content(res, as = "text", encoding = "UTF-8")
  out = fromJSON(txt)$message
  
  if("thinking" %in% names(out) & verbose) cat("\nmodel thinking: ", out$thinking,"\n")
  out$content
  }


# finish

cat("\n\n", text_gen("Say 'everything loaded and ready to go' in a funny way. Sign with your model name.", 100L), "\n\n")

