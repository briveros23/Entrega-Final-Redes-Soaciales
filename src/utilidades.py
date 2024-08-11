import igraph as ig
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objs as go
import math
from itertools import combinations
import nltk
from collections import Counter, defaultdict
import pandas as pd
from wordcloud import WordCloud



### Funciones para crear grafos bipartitos

def grafo_bipartito(primer_grupo, segundo_grupo, aristas):
    # nodos 
    nodos = primer_grupo + segundo_grupo
    # lista de True por ley
    primer_tipo = [True] * len(primer_grupo)
    # lista de false por partido politico 
    segundo_tipo = [False] * len(segundo_grupo)
    # sumamos ambas listas 
    tipo = primer_tipo + segundo_tipo  
    # creamos el grafo bipartito

    conexiones_indices = [(nodos.index(nodo1), nodos.index(nodo2)) for nodo1,nodo2  in aristas]
    
    g = ig.Graph.Bipartite(types=tipo, edges=conexiones_indices)
    g.vs['nombre'] = nodos

    return g

def generar_proyeccion(bipartite_graph, nom_variable = 'name'):
    # Verificar que el grafo sea bipartito
    if not bipartite_graph.is_bipartite():
        raise ValueError("El grafo proporcionado no es bipartito.")
    
    # Obtener el atributo tipo para determinar las dos particiones de nodos
    types = bipartite_graph.vs["type"]
    
    # Crear las proyecciones
    projection1 = bipartite_graph.bipartite_projection(which=0)
    projection2 = bipartite_graph.bipartite_projection(which=1)
    
    # Asignar los nombres a las proyecciones
    projection1.vs[nom_variable] = [bipartite_graph.vs[idx][nom_variable] for idx in range(len(types)) if types[idx] == 0]
    projection2.vs[nom_variable] = [bipartite_graph.vs[idx][nom_variable] for idx in range(len(types)) if types[idx] == 1]
    
    return projection1, projection2

### Funciones para graficar

def graficar_grafo(grafo, layout=None, path = None,titulo = None, **kwargs):
    if layout is None:
        layout = grafo.layout("kk")
    if titulo is None:
        titulo = "Grafo"
    plt.title(titulo, fontsize=15)
    plt.axis('off')  # Quitar los ejes
    ig.plot(grafo,target = path ,layout=layout, format = "png", **kwargs)
    print('Grafo guardado en:', path)


def generargrafico(g,titulo,layout):
    # Obtener los grados de los nodos
    degrees = g.degree()

    # Asignar colores a cada grupo (usando rojo y azul en este caso)
    node_colors = ['red' if g.vs[i]['type'] == 0 else 'blue' for i in range(g.vcount())]

    # Calcular el tamaño de los nodos basado en el grado (más grande para nodos con mayor grado)
    node_sizes = [5 + math.log(degrees[i]) for i in range(g.vcount())]

    # Obtener las posiciones de los nodos para el layout
    layout = g.layout(layout)

    # Obtener las coordenadas x e y de los nodos
    x_coords = [layout[x][0] for x in range(len(layout))]
    y_coords = [layout[x][1] for x in range(len(layout))]

    # Crear los nodos
    node_trace = go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers',
        marker=dict(size=node_sizes, color=node_colors, colorscale='Viridis'),  # Usar números como etiquetas de los nodos
        text = g.vs['name'],
        hoverinfo='text',
        
    )

    # Crear las aristas
    edge_trace = []
    for edge in g.get_edgelist():
        x0, y0 = layout[edge[0]]
        x1, y1 = layout[edge[1]]
        edge_trace.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(width=1, color='gray')
        ))

    # Crear el layout y la figura
    layout = go.Layout(
        title=titulo,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    fig = go.Figure(data=edge_trace + [node_trace], layout=layout)


    return fig 

def grafo_matriz_adyacencia(matriz_adyacencia, nombres_nodos):
    G = ig.Graph.Adjacency((matriz_adyacencia > 0).tolist(), mode = "undirected")
    G.es['weight'] = matriz_adyacencia[matriz_adyacencia.nonzero()]
    G.vs['name'] = nombres_nodos
    return G


def estadisticas_descriptivas(G):
    # Calculate centrality measures and add them as vertex attributes
    G.vs["closeness"] = G.closeness()
    G.vs["betweenness"] = G.betweenness()
    G.vs["eigen"] = G.eigenvector_centrality()
    strengths = G.strength(weights='weight')
    max_strength = max(strengths)
    max_strength_node_index = strengths.index(max_strength)
    
    return {
        # Caracterizar vertices del grafo
        # Diametro del grafo
        'diametro': G.diameter(),
        # Nodo de mayor grado
        'nodo_mayor_grado': G.vs.select(_degree=G.maxdegree())['name'],
        # mayor grado:
        'grado_mayor': G.maxdegree(),
        # nodo de mayor fuerza:
        'nodo_mayor_fuerza': G.vs[max_strength_node_index]['name'],
        # mayor fuerza:
        'fuerza_mayor': max_strength,
        # Nodo con mayor centralidad closeness
        'nodo_centralidad_closeness': G.vs.select(closeness=max(G.vs["closeness"]))[0]['name'],
        # mayor centralidad closeness
        'centralidad_closeness_mayor': max(G.vs["closeness"]),
        # Nodo con mayor centralidad betweenness
        'nodo_centralidad_betweenness': G.vs.select(betweenness=max(G.vs["betweenness"]))[0]['name'],
        # mayor centralidad betweenness
        'centralidad_betweenness_mayor': max(G.vs["betweenness"]),
        # Nodo con mayor centralidad propia
        'nodo_centralidad_eigen': G.vs.select(eigen=max(G.vs["eigen"]))[0]['name'],
        # mayor centralidad propia
        'centralidad_eigen_mayor': max(G.vs["eigen"]),
        # Caracterizar conectividad del grafo
        # Grado promedio
        'grado_promedio': sum(G.degree()) / len(G.degree()),
        # Clan mas grande
        'clan_mas_grande': len(G.largest_cliques()[0]),
        # Densidad de la red
        'densidad': G.density(),
        # transitividad local
        'transitivity_Global': G.transitivity_undirected(),
    }

def graficar_distribucion_grado(grafos, nombres_grafos=None):
    fig = go.Figure()

    # Si no se proporcionan nombres para los grafos, se usan nombres por defecto
    if nombres_grafos is None:
        nombres_grafos = [f'Grafo {i+1}' for i in range(len(grafos))]

    # Iterar sobre cada grafo en la lista de grafos
    for i, grafo in enumerate(grafos):
        # Calcular el grado de los nodos para el grafo actual
        grados = grafo.degree()
        
        # Agregar un histograma para el grafo actual
        fig.add_trace(go.Histogram(x=grados, name=nombres_grafos[i], opacity=0.75))

    # Añadir título y etiquetas
    fig.update_layout(
        title="Distribución del Grado de Grafos",
        xaxis_title="Grado",
        yaxis_title="Frecuencia",
        legend_title="Grafo",
        barmode='overlay'
    )

    return fig

def Generacion_de_skipgramas(text, n_palabras, k_saltos):
    '''Genera skipgrams de un texto dado con un tamaño de ventana n y un número de skips k.'''
    # Tokenize words
    words = nltk.word_tokenize(text)
    
    # Initialize list to store skipgrams
    skipgrams_list = []
    
    # Generate skipgrams
    for i in range(len(words)):
        # Create a window for the current position
        window = words[i:i+n_palabras+k_saltos]
        if len(window) < n_palabras:
            continue
        
        # Generate skipgrams from the window
        for j in range(len(window)):
            for k in range(j+1, min(j+1+k_saltos+1, len(window))):
                skipgram = (window[j], window[k])
                skipgrams_list.append(skipgram)
    skipgrams_list_final=[]
    for i in range(len(skipgrams_list)-1):
        if skipgrams_list[i]!=skipgrams_list[i+1]:
            skipgrams_list_final.append(skipgrams_list[i])
    return skipgrams_list_final

def plot_bigramas(skipgramas,top_n=10):
    # Conteo de bigramas
    conteo_bigramas = Counter(skipgramas)

    # Obtener los 10 bigramas más frecuentes
    bigramas_mas_frecuentes = conteo_bigramas.most_common(top_n)

    # Obtener datos para el gráfico
    bigramas_unicos = [bigrama[0] for bigrama in bigramas_mas_frecuentes]
    frecuencias = [frecuencia for _, frecuencia in bigramas_mas_frecuentes]

    # Graficar
   
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(bigramas_unicos)), frecuencias, tick_label=[f"{bigrama[0]} {bigrama[1]}" for bigrama in bigramas_unicos])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Bigramas')
    plt.ylabel('Frecuencia')
    plt.title('Top 10 Bigramas más Frecuentes')
    plt.show()
    return plt

def creacion_del_grafo(aristas):
    # Crear un grafo
    G = ig.Graph()

    # Conjunto de nodos existentes
    nodos_existentes = set()

    # Añadir las aristas al grafo
    for arista, peso in aristas:
        nodo1, nodo2 = arista

        if nodo1 != nodo2: 

            # Verificar si los nodos ya existen
            if nodo1 not in nodos_existentes:
                G.add_vertex(nodo1)  # Agregar vértice origen
                nodos_existentes.add(nodo1)  # Agregar nodo al conjunto de nodos existentes
            if nodo2 not in nodos_existentes:
                G.add_vertex(nodo2)  # Agregar vértice destino
                nodos_existentes.add(nodo2)  # Agregar nodo al conjunto de nodos existentes

            # Agregar arista con peso
            G.add_edge(nodo1, nodo2, weight=peso)

    return G

def bigramas_para_grafo(lista_bigramas, umbral=10):
    # Utilizamos Counter para contar las ocurrencias de las tuplas en la lista
    lista_bigramas = [tuple(sorted(tupla)) for tupla in lista_bigramas]
    contador = Counter(lista_bigramas)
    
    # Filtramos las tuplas que tienen una frecuencia mayor que el umbral
    tuplas_filtradas = [(tupla ,frecuencia)for tupla, frecuencia in contador.items() if frecuencia > umbral]
    
    return tuplas_filtradas


def frecuencia_de_las_palabras(sentences_dict):
    # Crear un diccionario para almacenar los contadores de palabras por nombre
    word_counts = {name: Counter(' '.join(sentences).split()) for name, sentences in sentences_dict.items()}
    
    # Obtener el total de palabras por nombre
    total_words = {name: sum(counter.values()) for name, counter in word_counts.items()}
    
    # Obtener todas las palabras únicas
    all_words = set(word for counter in word_counts.values() for word in counter)
    
    # Crear una lista para almacenar las filas del DataFrame
    data = []
    
    # Recorrer todas las palabras únicas
    for word in all_words:
        row = {'palabra': word}
        for name in sentences_dict.keys():
            # Calcular la frecuencia relativa
            row[f'frecuencia_relativa_{name}'] = word_counts[name][word] / total_words[name]
        data.append(row)
    
    # Crear el DataFrame
    df = pd.DataFrame(data).fillna(0).sort_values(by='palabra').reset_index(drop=True)
    
    return df

def generar_wordcloud(df, name_column):
    word_freq = dict(zip(df['palabra'], df[name_column]))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Nube de Palabras para {name_column}')
    plt.show()

def cluster_seleccion_componenteconexa_gigante(G, path_conexo_gigante,path_cluster):
    # Calcular la componente conexa gigante
    G_gigante = G.connected_components().giant()
    layout = G_gigante.layout("kk")
    # calcular la suma de los pesos de las aristas para cada nodo
    sum_pesos_aristas = G_gigante.strength(weights='weight')
    # Normalizar la suma de los pesos de las aristas para ajustarla al rango de tamaños de los nodos
    max_sum_pesos_aristas = max(sum_pesos_aristas)
    sizes = [x / max_sum_pesos_aristas * 40 for x in sum_pesos_aristas]  # Ajusta el rango de tamaño deseado
    G_gigante.vs["size"] = sizes
    ig.plot(G_gigante,path_conexo_gigante, layout=layout, bbox=(700, 700), margin=50,vertex_size=3,vertex_label_size=15,vertex_label_dist=2,edge_arrow_size=0.5,edge_width=0.5,vertex_label=G_gigante.vs['name'],vertex_color='lightblue',edge_color='gray')

    # Calcular la dendrograma de la comunidad
    dendrogram = G_gigante.community_edge_betweenness()

    # Obtener las comunidades finales
    communities = dendrogram.as_clustering()

    # Obtener el número de comunidades
    num_communities = len(communities)

    # Asignar colores a cada comunidad
    palette = ig.RainbowPalette(n=num_communities)
    community_colors = [palette.get(i) for i in communities.membership]

    # Definir transparencia para los nodos
    node_transparency = 0.5  # Valor entre 0 y 1, donde 0 es completamente transparente y 1 es completamente opaco

    # Convertir los colores de las comunidades a RGBA con la transparencia deseada
    community_colors_with_alpha = [color[:-1] + (node_transparency,) for color in community_colors]

    # Dibujar el grafo con nodos semi-transparentes
    ig.plot(G_gigante,path_cluster, layout=layout,vertex_color=community_colors_with_alpha)


    for community_id in range(num_communities):
        # Obtener los nodos de la comunidad actual
        community_nodes = [node for node, membership in enumerate(communities.membership) if membership == community_id]
        
        # Calcular el grado de cada nodo en la comunidad
        node_degrees = [G_gigante.degree(node) for node in community_nodes]
        
        # Encontrar el nodo con el mayor grado
        node_with_max_degree_index = node_degrees.index(max(node_degrees))
        node_with_max_degree = community_nodes[node_with_max_degree_index]
        
        # Obtener la palabra asociada al nodo con el mayor grado
        most_connected_word = G_gigante.vs[node_with_max_degree]['name']
        
        # Imprimir la palabra más importante de la comunidad actual
        print(f"Comunidad {community_id}: Palabra más importante (según grado): {most_connected_word}")


def create_graph_from_similarity_matrix(similarity_matrix, positive_threshold=0.5, negative_threshold=-0.5):
    # Lista de tuplas (i, j) para las aristas y sus pesos
    edges = []
    weights = []

    # Recorre la matriz de similitud para agregar aristas
    for i in range(similarity_matrix.shape[0]):
        for j in range(i + 1, similarity_matrix.shape[1]):
            weight = similarity_matrix[i, j]
            # Filtra aristas por los umbrales de similitud
            if weight > positive_threshold or weight < negative_threshold:
                edges.append((i, j))
                weights.append(weight)
    
    # Crear el grafo
    g = ig.Graph(edges=edges)
    
    # Añadir los pesos como atributo de las aristas
    g.es['weight'] = weights
    
    return g
