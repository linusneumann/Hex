import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


g1 = nx.grid_2d_graph(3,3)
# Initialisiere einen ungerichteten Graphen
board = nx.Graph()

# Spielfeldgröße (z.B. 3x3)
size = 3

# Erstelle Knoten und Kanten
for x in range(size):
    for y in range(size):
        board.add_node((x, y))  # Jeder Knoten repräsentiert ein Feld
        # Verbindungen zu benachbarten Feldern (oben, unten, links, rechts)
        if x > 0:
            board.add_edge((x, y), (x-1, y))
        if y > 0:
            board.add_edge((x, y), (x, y-1))

# Initialisiere alle Kanten mit einem Basisgewicht (z.B. 1 für freie Felder)
for edge in board.edges:
    board.edges[edge]['weight'] = 1  # Standardgewicht



# Wenn ein Spieler ein Feld beansprucht, passen wir die Gewichte an
def claim_field(player, node):
    for neighbor in board.neighbors(node):
        # Beispiel: Spieler 1 setzt Gewicht auf 0. Spieler 2 blockiert (hohes Gewicht).
        board.edges[node, neighbor]['weight'] = 0 if player == 1 else 10

def shortest_path(graph, start_nodes, end_nodes):
    # Finde den kürzesten Pfad zwischen zwei Mengen von Knoten
    min_distance = float('inf')
    for start in start_nodes:
        for end in end_nodes:
            try:
                distance = nx.shortest_path_length(graph, source=start, target=end, weight='weight')
                min_distance = min(min_distance, distance)
            except nx.NetworkXNoPath:
                continue
    return min_distance

# Seiten des Spielfelds für Spieler 1 (oben -> unten)
player1_start = [(0, y) for y in range(size)]
player1_end = [(size-1, y) for y in range(size)]

# Kürzeste Distanz für Spieler 1
distance_player1 = shortest_path(board, player1_start, player1_end)
print("Kürzeste Distanz Spieler 1:", distance_player1)

def best_move(graph, player, player_start, player_end):
    best_move = None
    best_score = float('inf')
    
    for node in graph.nodes:
        if not any(graph.edges[node, neighbor]['weight'] == 0 for neighbor in graph.neighbors(node)):
            # Simuliere Zug
            original_weights = {edge: graph.edges[edge]['weight'] for edge in graph.edges}
            claim_field(player, node)
            
            # Bewertung basierend auf kürzestem Pfad
            score = shortest_path(graph, player_start, player_end)
            
            # Rückgängig machen
            for edge, weight in original_weights.items():
                graph.edges[edge]['weight'] = weight
            
            if score < best_score:
                best_score = score
                best_move = node
    
    return best_move

next_move = best_move(board, player=1, player_start=player1_start, player_end=player1_end)
#print("Bester Zug für Spieler 1:", next_move)
#nx.draw(g1,with_labels=True)
#plt.show()

