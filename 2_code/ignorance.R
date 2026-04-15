source("2_code/_helpers.R")

ignor = read_csv("1_data/deliberate_ignorance.csv")

## Cleaning

# remove NAs; include articles & proceedings
exclude = is.na(ignor$`Author(s) ID`) | 
         is.na(ignor$References) |
         is.na(ignor$Title) | 
         is.na(ignor$Abstract) | ignor$Abstract == "[No abstract available]"
include = ignor$`Document Type` %in% c("Article", "Conference paper")
ignor = ignor |> filter(!exclude, include)

# clean abstracts
regex = c("Copyright[:print:]+$", "©[:print:]+$", "\\b[A-Z][a-z]+\\b\\:", "\\[(.*?)\\]")
for(reg in regex) ignor$Abstract = ignor$Abstract |> str_remove(reg) |> str_squish()

## Nets

### Authors

authors = str_split(ignor$`Author(s) ID` |> str_remove_all("[:blank:]"), ";")
author_tab = authors |> unlist() |> table()
author_tab = author_tab[author_tab > 1] # remove rare authors

author_mat = matrix(0, nrow = nrow(ignor), ncol = length(author_tab), 
                    dimnames = list(ignor$Title, names(author_tab)))
for(i in 1:nrow(ignor)) {
  authors_i = authors[[i]][authors[[i]] %in% names(author_tab)] 
  author_mat[i, authors_i] = 1
  }

svd = RSpectra::svds(author_mat, k = 30)
author_emb = svd$u %*% diag(svd$d)

### References

references = str_split(ignor$References |> str_replace_all("\\);", ")@@"), "@@ ")
reference_tab = references |> unlist() |> table()
reference_tab = reference_tab[reference_tab > 1] # remove rare authors

reference_mat = matrix(0, nrow = nrow(ignor), ncol = length(reference_tab), 
                       dimnames = list(ignor$Title, names(reference_tab)))
for(i in 1:nrow(ignor)) {
  references_i = references[[i]][references[[i]] %in% names(reference_tab)] 
  reference_mat[i, references_i] = 1
}

svd = RSpectra::svds(reference_mat, k = 30)
reference_emb = svd$u %*% diag(svd$d)

### Semantic

semantic_emb = qwen$encode(paste(ignor$Title, ". ", ignor$Abstract), show_progress_bar = TRUE)
saveRDS(semantic_emb, "1_data/intermediate/semantic_emb.RDS")
semantic_emb = readRDS("1_data/intermediate/semantic_emb.RDS")

### Combine

author_emb_norm = normalize(author_emb)
reference_emb_norm = normalize(reference_emb)
semantic_emb_norm = normalize(semantic_emb) # giving semantic a bit more weight

combined_emb = author_emb_norm |> 
  cbind(reference_emb_norm) |> 
  cbind(semantic_emb_norm)

combined_net = cosine(combined_emb)
rownames(combined_net) = colnames(combined_net) = ignor$Title

## Map & Clusters

umap.defaults$metric = "cosine"
umap.defaults$n_neighbors = 10
umap.defaults$min_dist = .3
layout = umap(combined_emb, config = umap.defaults)

ignor$x = layout$layout[,1]
ignor$y = layout$layout[,2]
ignor$cluster = hclust(dist(layout$layout), method = "ward.D2") |> cutree(10)

plot(ignor$x, ignor$y, col = ignor$cluster + 1)

## Tags

prompt = read_file("1_data/prompts/tag_prompt")
tags = character(length(prompt))
for(i in 1:nrow(ignor)){
  text = paste0("Title: ", ignor$Title[i], 
                "\nAbstract: ", ignor$Abstract[i])  
  tags[i] = text_gen(glue(prompt), len = 1000, model = "gemma4:e2b", think = TRUE)  # alternative: Qwen/Qwen3.5-0.8B 
  cat("Article ", i,": ", tags[i], "\n", sep = "")
  }

saveRDS(tags, "1_data/intermediate/tags.RDS")
tags = readRDS("1_data/intermediate/tags.RDS")

ignor$tags = str_remove_all(tags, "Tags=|\\[|\\]") |> str_split(";") |> 
  lapply(str_to_lower)

tag_tab = ignor$tags |> unlist() |> table()
tag_emb = qwen$encode(paste0("Research tag of delibarate ignorance article: ", names(tag_tab)), show_progress_bar = TRUE)
tag_cos = cosine(tag_emb)

cl = hclust(as.dist(1 - tag_cos), method = "complete")
tag_dict = tibble(tag = names(tag_tab), 
                  n = c(tag_tab),
                  cl = cutree(cl, min(which(cl$height>.07)))) |> 
  group_by(cl) |> 
  mutate(label = tag[which.max(n)]) |> 
  pull(label, tag)

ignor$tags_clean = lapply(ignor$tags, function(x) tag_dict[x])

## Labels

get_top = function(x){
  tab = x |> unlist() |> table() |> sort(decreasing = T)
  paste0(names(tab)[1:30], " (p = ", round(c(tab)[1:30] / sum(tab), 2), ")") |> 
    paste(collapse="\n")
  }
top_twenty_tags = split(ignor$tags_clean, ignor$cluster) |> sapply(get_top)

text = paste0("Cluster ", 1:10, ":\n", top_twenty_tags, "\n\n") |> paste(collapse="")

prompt = read_file("1_data/prompts/label_prompt")
labels = text_gen(glue(prompt), len = 5000, model = "gemma4:e2b", think = TRUE)
labels = labels |> str_remove_all("Labels=|\\[|\\]") |> str_split(";") |> unlist()
labels

## Visaualize


### landscape
cols = mako(max(ignor$cluster), begin = .1, end = .9)

par(mar=c(0,0,0,0),mfrow=c(1,1))
plot.new();plot.window(xlim = range(ignor$x), ylim = range(ignor$y))
points(ignor$x, ignor$y, col = cols[ignor$cluster], pch = 16)

c_x = tapply(ignor$x, ignor$cluster, mean)
c_y = tapply(ignor$y, ignor$cluster, mean)
pos_x = c_x + c(1.2,0,1.5,0,-1.3,1.5,0,0,0,.5)
pos_y = c_y + c(0,1.7,-1.5,-1.3,-.4,0,2,-1.5,-1.5,0)
text(pos_x, pos_y, labels = labels |> str_replace(" ","\n"), 
     col = cols, font=2, cex=1.2)



### tag inlandscape

top_tags = (ignor$tags_clean |> unlist() |> table() |> sort(decreasing = T) |> names())[1:40]

par(mfrow = c(5,8), mar=c(.5, .5, 2, .5))
for(i in 1:length(top_tags)){
  present = sapply(ignor$tags_clean, function(x) top_tags[i] %in% x)
  col = ifelse(present, "red", "black")
  plot.new();plot.window(xlim = range(ignor$x), ylim = range(ignor$y))
  points(ignor |> filter(!present) |> select(x, y), col = "grey75", pch=16, cex=.5)
  points(ignor |> filter(present) |> select(x, y), col = "red", pch=16, cex=.7)
  mtext(top_tags[i], cex=.5)
  }


### cluster relatedness

average_by_cluster = function(m, cl){
  
  mat = matrix(nrow = max(cl), ncol = max(cl))
    for(i in 1:max(cl)){
      for(j in i:max(cl)){
        mat[i,j] = mat[j,i] = mean(m[cl = cl[i], cl = cl[j]])
      }
    }
  mat}

author_net = cosine(author_emb_norm)
reference_net = cosine(reference_emb_norm)
sematic_net = cosine(semantic_emb_norm)

average_by_cluster(author_net, ignor$cluster)
average_by_cluster(sematic_net, ignor$cluster)










