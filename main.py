import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tkinter as tk
from queue import PriorityQueue
from tkinter import ttk


def criar_galpao(linhas, colunas, prateleiras_linhas, prateleiras_colunas):
    galpao = np.zeros((linhas, colunas), dtype=int)

    for linha in prateleiras_linhas:
        for coluna in prateleiras_colunas:
            if coluna < colunas:
                galpao[linha, coluna] = 1  # 1 representa uma prateleira

    galpao[3, 12] = 2  # 2 representa um robô

    paredes = [(1, 2), (1, 4), (1, 10), (2, 3), (2, 7), (4, 3), (4, 7), (5, 2),
               (5, 4), (5, 10)]

    for parede in paredes:
        galpao[parede] = 3  # 3 representa uma parede

    return galpao


def plotar_galpao(ax, galpao, labels, caminho=[]):
    cmap = mcolors.ListedColormap(['white', 'blue', 'green', 'black'])
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(galpao, cmap=cmap, norm=norm, interpolation='none')

    ax.set_xticks(np.arange(-.5, galpao.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, galpao.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

    ax.tick_params(which='both',
                   bottom=False,
                   left=False,
                   labelbottom=False,
                   labelleft=False)
    ax.set_title('Grid do Galpão')

    for (i, j), label in labels.items():
        ax.text(j, i, label, ha='center', va='center', color='black')

    # Plotar o caminho encontrado
    for (i, j) in caminho:
        ax.add_patch(
            plt.Rectangle((j - 0.5, i - 0.5),
                          1,
                          1,
                          fill=None,
                          edgecolor='red',
                          linewidth=2))


def criar_grafo(galpao, labels):
    linhas, colunas = galpao.shape
    G = nx.Graph()

    for linha in range(linhas):
        for coluna in range(colunas):
            if galpao[linha, coluna] in [0, 2]:
                G.add_node(labels[(linha, coluna)])

                if linha > 0 and galpao[linha - 1, coluna] in [0, 2]:
                    G.add_edge(labels[(linha, coluna)],
                               labels[(linha - 1, coluna)])
                if linha < linhas - 1 and galpao[linha + 1, coluna] in [0, 2]:
                    G.add_edge(labels[(linha, coluna)],
                               labels[(linha + 1, coluna)])
                if coluna > 0 and galpao[linha, coluna - 1] in [0, 2]:
                    G.add_edge(labels[(linha, coluna)],
                               labels[(linha, coluna - 1)])
                if coluna < colunas - 1 and galpao[linha,
                                                   coluna + 1] in [0, 2]:
                    G.add_edge(labels[(linha, coluna)],
                               labels[(linha, coluna + 1)])

    return G


def plotar_grafo(ax, G, pos, caminho=[]):
    nx.draw(G,
            pos,
            ax=ax,
            with_labels=True,
            node_size=300,
            node_color='lightblue',
            font_size=8,
            font_weight='bold')
    ax.set_title('Grafo dos Caminhos')

    # Plotar o caminho encontrado
    if caminho:
        edges = [(caminho[i], caminho[i + 1]) for i in range(len(caminho) - 2)]
        nx.draw_networkx_edges(G,
                               pos,
                               edgelist=edges,
                               ax=ax,
                               edge_color='red',
                               width=2)


def heuristica_manhattan(a, b):
    if heuristica_nao_admissivel.get():
        return (abs(a[0] - b[0]) + abs(a[1] - b[1])) * 2
    else:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(galpao, start, goal):
    linhas, colunas = galpao.shape
    start = (start[0], start[1])
    goal = (goal[0], goal[1])

    open_set = PriorityQueue()
    open_set.put((0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristica_manhattan(start, goal)}

    open_list = [start]
    closed_list = []

    # Registrar os estados das listas de nós abertos e fechados
    steps = [([], open_list.copy())]

    while not open_set.empty():
        current = open_set.get()[1]
        open_list.remove(current)
        closed_list.append(current)

        if current == goal:
            caminho = []
            while current in came_from:
                caminho.append(current)
                current = came_from[current]
            caminho.append(start)
            return caminho[::-1], steps

        for d in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + d[0], current[1] + d[1])
            if 0 <= neighbor[0] < linhas and 0 <= neighbor[
                    1] < colunas and galpao[neighbor] != 3:
                # Verificar se o vizinho é uma prateleira e se está bloqueando o caminho
                if galpao[neighbor] == 1 and neighbor != goal:
                    continue

                temp_g_score = g_score[current] + 1

                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + heuristica_manhattan(
                        neighbor, goal)
                    if neighbor not in open_list:
                        open_set.put((f_score[neighbor], neighbor))
                        open_list.append(neighbor)

        # Registrar o estado atual das listas cumulativas
        steps.append((closed_list.copy(), open_list.copy()))

    return None, steps


def ir_para_destino():
    destino = combo_destinos.get()
    print("Destino selecionado:", destino)

    destino_coords = None
    for (i, j), label in labels.items():
        if label == destino:
            destino_coords = (i, j)
            print("Coord. destino selecionado:", destino_coords)
            break

    start_coords = (3, 12)
    caminho, steps = a_star(galpao, start_coords, destino_coords)

    if caminho:
        print("Caminho encontrado:", caminho)
        fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(12, 6))
        plotar_galpao(ax1, galpao, labels, caminho)
        plotar_grafo(ax2, grafo, pos, [labels[(i, j)] for (i, j) in caminho])

        # Criar a tabela de nós abertos e fechados ao longo dos passos
        tabela_data = []
        for closed, open_ in steps:
            open_labels = [labels[n] for n in open_]
            closed_labels = [labels[n] for n in closed]

            # Remover duplicatas e ordenar as labels
            open_labels = sorted(set(open_labels), key=open_labels.index)
            closed_labels = sorted(set(closed_labels), key=closed_labels.index)

            # Formatando as strings sem vírgulas extras
            open_str = ', '.join(open_labels) if open_labels else '-'
            closed_str = ', '.join(closed_labels) if closed_labels else '-'

            tabela_data.append((open_str, closed_str))

        tabela = ax3.table(cellText=tabela_data,
                           colLabels=['Nós Abertos', 'Nós Fechados'],
                           cellLoc='center',
                           loc='center')
        tabela.auto_set_font_size(False)
        tabela.set_fontsize(10)

        # Ajustar a posição da tabela para criar uma margem com o título
        tabela.auto_set_column_width(
            col=list(range(len(['Nós Abertos', 'Nós Fechados']))))
        tabela.scale(
            0.8, 1.2
        )  # Ajuste a largura e a altura da tabela novamente, se necessário

        ax3.axis('off')
        ax3.set_title('Listas de Nós Abertos e Fechados',
                      pad=20)  # Adicionar margem com pad

        # Deixar o quarto subplot vazio
        ax4.axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("Nenhum caminho encontrado.")


# Configurações do galpão
linhas = 7
colunas = 13

# Posicionar as prateleiras
prateleiras_linhas = [0, 2, 4, 6]
prateleiras_colunas = [0, 2, 4, 6, 8, 10, 12]

# Criar o galpão
galpao = criar_galpao(linhas, colunas, prateleiras_linhas, prateleiras_colunas)

# Criar labels para cada posição
labels = {
    (i, j): f'P{i * colunas + j}'
    for i in range(linhas)
    for j in range(colunas)
}

# Mapear as siglas de volta para coordenadas para o plot do grafo
pos = {labels[(i, j)]: (j, -i) for i in range(linhas) for j in range(colunas)}

# Criar o grafo dos caminhos possíveis
grafo = criar_grafo(galpao, labels)

# Criar a janela principal
root = tk.Tk()
root.title("Seleção de Destino")

# Criar a lista de prateleiras como opções
prateleiras = [(linha, coluna) for linha in prateleiras_linhas
               for coluna in prateleiras_colunas]

# Mapear as prateleiras para suas siglas correspondentes
prateleiras_labels = {
    prateleira: f'P{prateleira[0] * colunas + prateleira[1]}'
    for prateleira in prateleiras
}

# Criar a combobox com as opções das prateleiras
combo_destinos = ttk.Combobox(root,
                              values=list(prateleiras_labels.values()),
                              state="readonly")
combo_destinos.pack(pady=10)

# Criar e posicionar a checkbox para ativar/desativar a exibição do caminho
heuristica_nao_admissivel = tk.BooleanVar()
chk_exibir_caminho = ttk.Checkbutton(root,
                                     text="Heurística Não Admissível",
                                     variable=heuristica_nao_admissivel)
chk_exibir_caminho.pack(pady=5)

# Criar e posicionar o botão para confirmar o destino
btn_confirmar = ttk.Button(root,
                           text="Confirmar Destino",
                           command=ir_para_destino)
btn_confirmar.pack(pady=5)

# Criar uma figura com quatro subplots (2x2)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6))

# Plotar o grid do galpão no primeiro subplot
plotar_galpao(ax1, galpao, labels)

# Plotar o grafo dos caminhos possíveis no segundo subplot
plotar_grafo(ax3, grafo, pos)

# Configurar o terceiro subplot para a tabela de nós abertos e fechados
ax2.axis('off')
ax2.set_title('Listas de Nós Abertos e Fechados')

# Deixar o quarto subplot vazio
ax4.axis('off')

# Mostrar a figura
plt.tight_layout()
plt.show()

# Iniciar o loop da aplicação
root.mainloop()
