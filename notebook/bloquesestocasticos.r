library(igraph)
library(blockmodels)
setwd('C:/Users/bcriv/OneDrive/Documentos/redes_sociales/Proyecto_final_redes_sociales/notebook')
# Importar el grafo desde el archivo GraphML
g <- igraph::read_graph("../input/graph_similaridad.graphml", format = "graphml")

# Ver el grafo importado
print(g)
adj_matrix <- as.matrix(as_adjacency_matrix(g,attr = 'weight'))
adj_matrix
unique(V(g)$partido)
########## visualizacion del grafico
# Asignar números a cada partido
party.nums <- as.numeric(as.factor(V(g)$partido))

# Definir una paleta de colores
cols <- RColorBrewer::brewer.pal(n = length(unique(party.nums)), name = "Set1")

# Graficar el grafo
par(mfrow = c(1,1), mar = 0.2*c(1,1,1,1))
set.seed(42)
plot(g, layout = layout_with_fr, vertex.label = NA, vertex.size = 4, 
     vertex.color = cols[party.nums], vertex.frame.color = cols[party.nums], 
     edge.color = adjustcolor("black", 0.1))
########## modelamiento 
sbm_model <- BM_gaussian("SBM", adj_matrix)
sbm_model$estimate()


ICLs <- sbm_model$ICL
num_blocks <- which.max(sbm_model$ICL)

#graficamos el icl

par(mfrow = c(1,1), mar = c(2.75,2.75,1.5,0.5), mgp = c(1.7,0.7,0))
plot(sbm_model$ICL, xlab = "Q", ylab = "ICL", type = "b", pch = 16)
lines(x = c(num_blocks,num_blocks), y = c(min(ICLs),max(ICLs)), col = "red", lty = 2)


# probabilidades de pertenercer a las comunidades


Z <- sbm_model$memberships[[num_blocks]]$Z
dim(Z)
labs <- apply(X = Z, MARGIN = 1, FUN = which.max)
head(x = labs, n = 10)
table(labs)
alpha <- table(labs)/vcount(g)
round(alpha[order(alpha, decreasing = T)], 3)
mu <- sbm_model$model_parameters[[num_blocks]]$mu
round(mu, 3)
corrplot::corrplot(corr = mu, type = "full", col.lim = c(0,1),  method = "shade", addgrid.col = "gray90", tl.col = "black")


###### funcione y resultados 

get_adjacency_ordered <- function(xi, A) {
  indices <- order(xi)
  xi_sorted <- xi[indices]
  
  # Encuentra los puntos donde cambia el valor en xi_sorted
  d <- which(diff(xi_sorted) != 0)
  
  # Reorganiza la matriz de adyacencia
  list(A = A[indices, indices], d = d)
}

# para graficar la matriz de adyacencia
heat.plot0 <- function(mat, show.grid = FALSE, cex.axis = 1, tick = TRUE, labs = NULL, col.axis = NULL, ...) {
  JJ <- nrow(mat)
  
  # Escala de colores
  colorscale <- c("white", rev(heat.colors(100)))
  color_range <- seq(floor(100 * min(mat)), floor(100 * max(mat)))
  
  # Definición de etiquetas y colores si faltan
  labs <- if (is.null(labs)) 1:JJ else labs
  col.axis <- if (is.null(col.axis)) rep("black", JJ) else col.axis
  
  # Graficar la matriz de adyacencia
  image(seq_len(JJ), seq_len(JJ), mat, axes = FALSE, xlab = "", ylab = "", col = colorscale[color_range], ...)
  
  # Añadir los ejes
  apply(matrix(seq_len(JJ)), 1, function(j) {
    axis(1, at = j, labels = labs[j], las = 2, cex.axis = cex.axis, tick = tick, col.axis = col.axis[j], col.ticks = col.axis[j])
    axis(2, at = j, labels = labs[j], las = 2, cex.axis = cex.axis, tick = tick, col.axis = col.axis[j], col.ticks = col.axis[j])
  })
  
  box()
  
  # Añadir la cuadrícula si es necesario
  if (show.grid) grid(nx = JJ, ny = JJ)
}


###### ordenacion
# asignaciones de grupos
xi <- apply(X = Z, MARGIN = 1, FUN = which.max)
# matriz de adyacencia original
Y <- adj_matrix
# matriz de adyacencia ordenada y lineas divisorias de acuerdo con las comunidades
tmp <- get_adjacency_ordered(xi = xi, A = Y)
# viz
par(mfrow = c(1,2), mar = 0.2*c(1,1,1,1))
# G
cols <- RColorBrewer::brewer.pal(n = 12, name = "Paired")
set.seed(42)
plot(g, layout = layout_with_fr, vertex.label = NA, vertex.size = 5, vertex.color = cols[labs], vertex.frame.color = cols[labs], edge.color = adjustcolor("black",0.1))
# A
heat.plot0(mat = tmp$A, tick = F, labs = NA)
abline(v = tmp$d+.5, h = tmp$d+.5)

###### comparacion entre los partidos y la nueva particion 

partidos <- as.numeric(as.factor(V(g)$partido))

agrpaest <- apply(X = Z, MARGIN = 1, FUN = which.max)

round(igraph::compare(comm1 = partidos, comm2 = agrpaest, method = "rand"), 4)
round(igraph::compare(comm1 = partidos, comm2 = agrpaest, method = "adjusted.rand"), 4)
round(igraph::compare(comm1 = partidos, comm2 = agrpaest, method = "nmi"), 4)

table(party.nums, labs)
