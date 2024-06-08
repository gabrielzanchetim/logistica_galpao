import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import networkx as nx
from heapq import heappop, heappush
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

# Definição do galpão
warehouse = [
    ['A1', 'L', 'B1', 'L', 'C1', 'L', 'D1', 'L', 'E1', 'L', 'F1', 'L', 'G1'],
    ['L', 'L', 'B', 'L', 'B', 'L', 'L', 'L', 'L', 'L', 'L', 'B', 'L'],
    ['A2', 'L', 'B2', 'B', 'C2', 'L', 'D2', 'B', 'E2', 'L', 'F2', 'L', 'G2'],
    ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'E'],
    ['A3', 'L', 'B3', 'B', 'C3', 'L', 'D3', 'B', 'E3', 'L', 'F3', 'L', 'G3'],
    ['L', 'L', 'B', 'L', 'B', 'L', 'L', 'L', 'L', 'L', 'B', 'L', 'L'],
    ['A4', 'L', 'B4', 'L', 'C4', 'L', 'D4', 'L', 'E4', 'L', 'F4', 'L', 'G4']
]


# Função para visualizar o galpão
def plot_warehouse(warehouse, robot_path=None, goal_pos=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    ax.clear()
    ax.set_xticks(np.arange(len(warehouse[0])) + 0.5, minor=True)
    ax.set_yticks(np.arange(len(warehouse)) + 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)

    ax.set_xlim(-0.5, len(warehouse[0]) - 0.5)
    ax.set_ylim(-0.5, len(warehouse) - 0.5)

    for y in range(len(warehouse)):
        for x in range(len(warehouse[0])):
            element = warehouse[y][x]
            if element == 'L':
                ax.text(x, y, element, ha='center', va='center', fontsize=12)
            elif element == 'B':
                ax.add_patch(
                    plt.Rectangle((x - 0.5, y - 0.5),
                                  1,
                                  1,
                                  fill=True,
                                  color="black"))
            elif element == 'E':
                ax.add_patch(
                    plt.Rectangle((x - 0.5, y - 0.5),
                                  1,
                                  1,
                                  fill=True,
                                  color="green"))
            elif (x, y) == goal_pos:
                package_img = mpimg.imread("package.png")
                ax.imshow(package_img,
                          extent=[x - 0.5, x + 0.5, y - 0.5, y + 0.5])
            else:
                ax.text(x,
                        y,
                        element,
                        ha='center',
                        va='center',
                        fontsize=12,
                        color='red')

    if robot_path:
        robo_img = mpimg.imread("robo.png")
        img_plot = ax.imshow(robo_img,
                             extent=[
                                 robot_path[0][0] - 0.5,
                                 robot_path[0][0] + 0.5,
                                 robot_path[0][1] - 0.5, robot_path[0][1] + 0.5
                             ])

        def update(frame):
            x, y = robot_path[frame]
            img_plot.set_extent([x - 0.5, x + 0.5, y - 0.5, y + 0.5])
            return img_plot,

        anim = FuncAnimation(fig1,
                             update,
                             frames=len(robot_path),
                             interval=300,
                             blit=True,
                             repeat=False)

        # Adiciona uma seta ao caminho percorrido
        for i in range(1, len(robot_path)):
            ax.annotate("",
                        xy=(robot_path[i][0], robot_path[i][1]),
                        xytext=(robot_path[i - 1][0], robot_path[i - 1][1]),
                        arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))

    ax.invert_yaxis()


# Função auxiliar para encontrar vizinhos válidos
def get_neighbors(pos, warehouse):
    neighbors = []
    x, y = pos
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(warehouse[0]) and 0 <= ny < len(
                warehouse) and warehouse[ny][nx] not in [
                    'B', 'A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'A2', 'B2',
                    'C2', 'D2', 'E2', 'F2', 'G2', 'A3', 'B3', 'C3', 'D3', 'E3',
                    'F3', 'G3', 'A4', 'B4', 'C4', 'D4', 'E4', 'F4', 'G4'
                ]:
            neighbors.append((nx, ny))
    return neighbors


# Função de heurística (distância de Manhattan)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Função para encontrar a posição adjacente válida mais próxima
def find_adjacent_goal(warehouse, goal, entry):
    x, y = goal
    possible_targets = [(x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y)]
    min_distance = float('inf')
    closest_target = None

    for target in possible_targets:
        if 0 <= target[0] < len(
                warehouse[0]) and 0 <= target[1] < len(warehouse):
            distance = abs(target[0] - entry[0]) + abs(target[1] - entry[1])
            if distance < min_distance:
                min_distance = distance
                closest_target = target

    if closest_target and warehouse[closest_target[1]][
            closest_target[0]] == 'L':
        return closest_target
    else:
        return goal  # Se não houver posição adjacente válida, retorna o próprio objetivo


# Função para encontrar a posição da entrada
def find_entry(warehouse):
    for y in range(len(warehouse)):
        for x in range(len(warehouse[0])):
            if warehouse[y][x] == 'E':
                return (x, y)
    return None


# Função para obter a posição de um objetivo dado seu rótulo
def find_goal_position(warehouse, goal_label):
    for y in range(len(warehouse)):
        for x in range(len(warehouse[0])):
            if warehouse[y][x] == goal_label:
                return (x, y)
    return None


# Função para plotar o grafo do caminho percorrido e salvar como imagem
def plot_graph(G, path, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 12))
    ax.clear()
    pos = {
        node: (node[1], -node[0])
        for node in G.nodes()
    }  # Inverte as coordenadas x e y
    nx.draw(G,
            pos,
            with_labels=True,
            node_size=500,
            node_color='lightblue',
            font_size=10,
            font_weight='bold',
            ax=ax)
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(G,
                           pos,
                           edgelist=path_edges,
                           edge_color='r',
                           width=2,
                           ax=ax)
    ax.invert_yaxis()  # Inverte o eixo y para desenhar de cima para baixo


# Função para plotar as listas Open e Closed como uma tabela e salvar como imagem
def plot_lists(open_list, closed_list, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    ax.clear()
    ax.axis('tight')
    ax.axis('off')
    table_data = []
    for open_item, closed_item in zip(open_list, closed_list):
        table_data.append([str(open_item), str(closed_item)])

    # Criar a tabela
    table = ax.table(cellText=table_data,
                     colLabels=['Open List', 'Closed List'],
                     loc='center',
                     cellLoc='center',
                     bbox=[0, 0, 1, 1])

    # Definir o tamanho da fonte
    table.auto_set_font_size(False)
    table.set_fontsize(8)  # Altere o valor para o tamanho desejado

    # Ajustar o tamanho das colunas
    table.auto_set_column_width([0, 1])


# Função para o algoritmo A*
def a_star_search(warehouse, start, goal):
    goal = find_adjacent_goal(warehouse, goal, start)
    start, goal = tuple(start), tuple(goal)
    open_list = []
    heappush(open_list, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    closed_list = set()  # Conjunto para armazenar os nós fechados

    open_list_data = [open_list.copy()]
    closed_list_data = [closed_list.copy()]

    # Grafo para visualizar o caminho percorrido
    G = nx.Graph()

    while open_list:
        _, current = heappop(open_list)
        closed_list.add(current)  # Adiciona o nó atual à lista fechada

        if current == goal:
            break

        for neighbor in get_neighbors(current, warehouse):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal, neighbor)
                heappush(open_list, (priority, neighbor))
                came_from[neighbor] = current
                G.add_edge(current, neighbor)  # Adiciona aresta ao grafo

        open_list_data.append(open_list.copy())
        closed_list_data.append(closed_list.copy())

    path = []
    current = goal
    while current:
        path.append(current)
        current = came_from[current]
    path.reverse()

    return path, G, open_list_data, closed_list_data


# Função para atualizar a GUI com os resultados da busca A*
def run_a_star():
    goal_label = entry_goal.get().upper()
    goal_pos = find_goal_position(warehouse, goal_label)

    if goal_pos is None:
        messagebox.showerror("Erro", "Destino não encontrado.")
        return

    start = find_entry(warehouse)
    path, G, open_list_data, closed_list_data = a_star_search(
        warehouse, start, goal_pos)

    plot_warehouse(warehouse, robot_path=path, goal_pos=goal_pos, ax=ax1)
    canvas1.draw()

    plot_graph(G, path, ax=ax2)
    canvas2.draw()

    plot_lists(open_list_data, closed_list_data, ax=ax3)
    canvas3.draw()


# Configuração da interface gráfica
root = tk.Tk()
root.title("Simulação de Robô no Galpão")
root.geometry("1600x900")

frame1 = ttk.Frame(root, width=800, height=450)
frame1.grid(row=0, column=0, sticky='nsew')
frame2 = ttk.Frame(root, width=800, height=450)
frame2.grid(row=0, column=1, sticky='nsew')
frame3 = ttk.Frame(root, width=800, height=450)
frame3.grid(row=1, column=0, sticky='nsew')
frame4 = ttk.Frame(root, width=800, height=450)
frame4.grid(row=1, column=1, sticky='nsew')

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Parte superior esquerda: desenho do galpão e caminho do robô
fig1, ax1 = plt.subplots(figsize=(8, 4))
canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Parte superior direita: grafo gerado
fig2, ax2 = plt.subplots(figsize=(8, 4))
canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Parte inferior esquerda: listas Open e Closed
fig3, ax3 = plt.subplots(figsize=(8, 4))
canvas3 = FigureCanvasTkAgg(fig3, master=frame3)
canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Parte inferior direita: pergunta de destino do robô, caixa de texto e botão de envio
label_goal = ttk.Label(
    frame4,
    text="Digite a casa para a qual deseja ir (por exemplo, A1, B1, etc.):")
label_goal.pack(pady=20)

entry_goal = ttk.Entry(frame4, width=20)
entry_goal.pack(pady=10)

button_submit = ttk.Button(frame4, text="Enviar", command=run_a_star)
button_submit.pack(pady=20)

root.mainloop()
