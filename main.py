import matplotlib.pyplot as plt
import numpy as np
from heapq import heappop, heappush
import itertools

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
def plot_warehouse(warehouse, paths=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xticks(np.arange(len(warehouse[0])) + 0.5, minor=True)
    ax.set_yticks(np.arange(len(warehouse)) + 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)

    # Ajuste do grid e elementos para evitar cortes
    ax.set_xlim(-0.5, len(warehouse[0]) - 0.5)
    ax.set_ylim(-0.5, len(warehouse) - 0.5)

    # Plotando os elementos do galpão
    for y in range(len(warehouse)):
        for x in range(len(warehouse[0])):
            element = warehouse[y][x]
            if element == 'L':
                ax.text(x, y, element, ha='center', va='center', fontsize=12)
            elif element == 'B':
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=True, color="black"))
            elif element == 'E':
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=True, color="green"))
            else:
                ax.text(x, y, element, ha='center', va='center', fontsize=12, color='red')

    # Plotando o caminho, se fornecido
    if paths:
        colors = ['blue', 'red', 'green']
        for idx, path in enumerate(paths):
            color = colors[idx % len(colors)]
            for i in range(len(path) - 1):
                (x1, y1), (x2, y2) = path[i], path[i + 1]
                ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.2, head_length=0.2, fc=color, ec=color)

    plt.gca().invert_yaxis()
    plt.show()

# Função auxiliar para encontrar vizinhos válidos
def get_neighbors(pos, warehouse):
    neighbors = []
    x, y = pos
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(warehouse[0]) and 0 <= ny < len(warehouse) and warehouse[ny][nx] != 'B':
            neighbors.append((nx, ny))
    return neighbors

# Função de heurística (distância de Manhattan)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Função para encontrar a posição adjacente válida mais próxima
def find_adjacent_goal(warehouse, goal):
    x, y = goal
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(warehouse[0]) and 0 <= ny < len(warehouse) and warehouse[ny][nx] == 'L':
            return (nx, ny)
    return goal  # Se não houver posição adjacente válida, retorna o próprio objetivo

# Implementação do algoritmo A*
def a_star_search(warehouse, start, goal):
    goal = find_adjacent_goal(warehouse, goal)
    start, goal = tuple(start), tuple(goal)
    open_list = []
    heappush(open_list, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while open_list:
        _, current = heappop(open_list)

        if current == goal:
            break

        for neighbor in get_neighbors(current, warehouse):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal, neighbor)
                heappush(open_list, (priority, neighbor))
                came_from[neighbor] = current

    path = []
    current = goal
    while current:
        path.append(current)
        current = came_from[current]
    path.reverse()

    return path

# Função para encontrar a posição da entrada
def find_entry(warehouse):
    for y in range(len(warehouse)):
        for x in range(len(warehouse[0])):
            if warehouse[y][x] == 'E':
                return (x, y)
    return None

# Função para calcular o caminho mínimo entre múltiplos destinos
def find_best_route(warehouse, start, goals):
    all_goals = [start] + goals
    best_path = []
    min_distance = float('inf')
    best_perm = None

    for perm in itertools.permutations(goals):
        current_path = []
        total_distance = 0
        current_position = start

        for goal in perm:
            path_segment = a_star_search(warehouse, current_position, goal)
            total_distance += len(path_segment) - 1
            current_path += path_segment[:-1] if current_path else path_segment
            current_position = goal

        if total_distance < min_distance:
            min_distance = total_distance
            best_path = current_path
            best_perm = perm

    if best_perm is None:
        best_perm = goals

    best_path.append(goals[-1])  # Adiciona a última posição da permutação
    return best_path, best_perm

# Função para exibir o caminho detalhado no console
def print_path_details(path, goals):
    goal_names = ['A1', 'B2', 'C3']  # Nomes das metas para exibição
    goal_positions = dict(zip(goal_names, goals))

    for i, goal in enumerate(goal_names):
        start_index = path.index(goal_positions[goal])
        end_index = path.index(goal_positions[goal]) + 1
        segment = path[start_index:end_index]
        segment_str = "->".join([f"({x},{y})" for (x, y) in segment])
        print(f"Caminho até {goal}: {segment_str}")

# Exemplo de uso para destinos A1, B2, C3
start = find_entry(warehouse)
goals = [(0, 0), (10, 0)]  # Posições de A1, B2, C3
best_path, best_perm = find_best_route(warehouse, start, goals)

# Verificando se best_perm foi encontrado corretamente
if best_perm is None:
    print("Nenhuma permutação válida foi encontrada.")
else:
    segments = []
    current_pos = start
    for goal in best_perm:
        segment = a_star_search(warehouse, current_pos, goal)
        segments.append(segment)
        current_pos = goal

    plot_warehouse(warehouse, segments)

    # Imprimindo o caminho detalhado no console
    print_path_details(best_path, goals)
