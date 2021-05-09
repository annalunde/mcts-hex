import copy
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Cell:

    def __init__(self, row, col):
        """
        A cell on the board
        Has neighbors, a filled_by indicator and a position: row, col
        """

        self.row = row
        self.col = col
        self.filled_by = 0  # Assuming cells are instantiated empty.
        self.neighbors = set()

    def fill(self, player_id):
        self.filled_by = player_id


class Hexboard:
    """
    NOTE: PLAYER ONE IS PLAYER 1 AND PLAYS HORIZONTALLY. PLAYER TWO: 2, PLAYS VERTICALLY.

    A Board instance permits a single gameplay episode and can only be played through once
    During training, the simulation environment will thus instantiate new boards iteratively

    """

    def __init__(self, size, first_player=1, track_history=False, init_state=None):
        """
        Initializes a Hexagonal board instance with start player 1 if first_player else 2


        Board indices [row,col] works as follows:
         row + 1 -> next row below
         col + 1 -> next element horizontally to the right

        Example of diamond board of size 3 illustrated below
                                                                    (0,0)
        [[ (0,0) , (0,1) , (0,2)  ]                            (1,0)     (0,1)
         [ (1,0) , (1,1) , (1,2)  ]       ->              (2,0)     (1,1)     (0,2)
         [ (2,0) , (2,1) , (2,2)  ]]                           (2,1)     (1,2)
                                                                    (2,2)
        :param size: Size of board
        :param first_player: If true, HORIZONTAL player, i.e. 1, goes first
        :param track_history: Whether or not to keep track of boards previous states during play
        :param init_state: State to initialize Hex board at
        """

        self.current_state = []  # Holds cell objects that keep track of their own neighbors
        self.size = size
        self.current_player_id = first_player
        self.track_history = track_history
        # Holds copies of (action,current_state,player) along the way
        self.history = [] if track_history else None

        # Build empty board with unfilled cell objects
        for row in range(size):
            r = []
            for col in range(size):
                r.append(Cell(row, col))
            self.current_state.append(r)

        # Ensure neighbor lists in cells are coherent
        for row in range(size):
            for col in range(size):
                potential_neighbors = [(row, col - 1), (row, col + 1),  # Horizontal
                                       (row - 1, col), (row + 1, col),  # Vertical
                                       (row + 1, col - 1), (row - 1, col + 1)]  # Diagonal
                # Find only those within board
                valids = filter(lambda coords: self.validate_coordinates((coords[0], coords[1])), potential_neighbors)
                c = self.current_state[row][col]
                for neighbor_row, neighbor_col in valids:
                    neighbor_cell = self.current_state[neighbor_row][neighbor_col]
                    c.neighbors.add(neighbor_cell)

        #  Update cell.filledby values if init state is given
        if init_state is not None:
            if first_player is None:  # Infer player_id from assumption that 1 plays every even move and 2 plays odd
                ones = init_state.count(1)
                twos = init_state.count(2)
                self.current_player_id = 1 if ones + twos % 2 == 0 else 2
            for row in range(size):
                for col in range(size):
                    self.current_state[row][col].filled_by = init_state[row * size + col]

        # Append snapshot of initial board state to history if tracking is ON
        if self.track_history:
            self.history.append((None, copy.deepcopy(self.current_state)))

    def get_actions(self):
        legal_actions = []
        # if not self.is_finished():
        for _ in self.current_state:
            for cell in _:
                if cell.filled_by == 0:
                    legal_actions.append((cell.row, cell.col))
        return legal_actions

    def generate_successor_states(self):
        legal_actions = self.get_actions()
        current_state_as_list, _ = self.get_state()
        successor_states = []
        for action in legal_actions:
            successor = current_state_as_list.copy()
            row, col = action[0], action[1]
            successor[row * self.size + col] = self.current_player_id
            successor_states.append(successor)

        # PLAYER SWITCH IS PERFORMED HERE AND ONLY HERE
        pid = 2 if self.current_player_id == 1 else 1
        return legal_actions, successor_states, pid

    def get_state(self):
        return self.convert_state_to_list(self.current_state), self.current_player_id

    def perform_action(self, action):
        if self.validate_coordinates(action):
            self.current_state[action[0]][action[1]].fill(
                self.current_player_id)
            if self.track_history:
                self.history.append(
                    (action, copy.deepcopy(self.current_state)))
            # We want to record the player that made the action before changing current player
            self.current_player_id = 1 if self.current_player_id == 2 else 2

    def convert_state_to_list(self, state):
        lst = []
        for row in state:
            for cell in row:
                lst.append(cell.filled_by)
        return lst

    def validate_coordinates(self, coords):
        # Checks whether index is within board
        for coord in coords:
            if coord < 0 or coord > self.size - 1:
                return False
        return True

    def is_finished(self, current_pid=None):
        if current_pid is None:
            pid = 1 if self.current_player_id == 2 else 2
        else:
            pid = 1 if current_pid == 2 else 2  # Winner has to be previous player
        djs = DisjSet(self.size ** 2 + 2)  # Add two for virtuall cells representing the edges

        # Create mapping from Cells to their idx in the disjoint set obj
        cell_to_idx = {}
        i = 0
        for row in self.current_state:
            for cell in row:
                cell_to_idx[cell] = i
                i += 1

        edge1 = Cell(None, None)  # Create virtual cells at edges
        edge2 = Cell(None, None)
        edge1.fill(pid)
        edge2.fill(pid)
        for j in range(self.size):
            # Add all cells at edges of board to neighbor list of edges
            if pid == 1:  # Then search horizontally
                edge1.neighbors.add(self.current_state[j][0])
                edge2.neighbors.add(self.current_state[j][self.size - 1])
            else:  # Then search vertically
                edge1.neighbors.add(self.current_state[0][j])
                edge2.neighbors.add(self.current_state[self.size - 1][j])

        cell_to_idx[edge1] = i
        cell_to_idx[edge2] = i + 1
        # First add all neighbors to edges groups
        for cell in edge1.neighbors:
            if cell.filled_by == pid:
                djs.Union(cell_to_idx[cell], cell_to_idx[edge1])
        for cell in edge2.neighbors:
            if cell.filled_by == pid:
                djs.Union(cell_to_idx[cell], cell_to_idx[edge2])

        # Then begin search from one side to the other
        for i in range(self.size - 1):
            for j in range(self.size):  # Don't need to loop over last column/row
                if pid == 1:  # Then we are iterating over columns with i, (and inside columns with j)
                    cell = self.current_state[j][i]
                else:  # Then we are iterating over rows with i, (and for each row loop over columns with j)
                    cell = self.current_state[i][j]
                if cell.filled_by == pid:
                    for neighbor in cell.neighbors:
                        if neighbor.filled_by == pid:
                            djs.Union(cell_to_idx[cell], cell_to_idx[neighbor])

        edge1_idx = cell_to_idx[edge1]
        edge2_idx = cell_to_idx[edge2]
        return djs.find(edge1_idx) == djs.find(edge2_idx)

    def dfs(self, visited, node, player_id):
        if node not in visited:
            visited.add(node)
            for neighbor in filter(lambda n: n.filled_by == player_id, node.neighbors):
                if player_id == 1:
                    if neighbor.col == self.size - 1:
                        return True
                else:
                    if neighbor.row == self.size - 1:
                        return True
                if self.dfs(visited, neighbor, player_id):
                    return True
            return False

    def is_finished_old(self):
        potential_roots = []
        potential_winner = 1 if self.current_player_id == 2 else 2
        if potential_winner == 1:  # The previous player
            # Need to check for HORIZONTAL path
            for row in range(self.size):
                potential = self.current_state[row][0]
                if potential.filled_by == potential_winner:
                    potential_roots.append(potential)
        else:
            # Need to check for VERTICAL path
            for col in range(self.size):
                potential = self.current_state[0][col]
                if potential.filled_by == potential_winner:
                    potential_roots.append(potential)

        # Perform DFS from potential roots and find out if any lead to other side
        visited = set()
        for p in potential_roots:
            if self.dfs(visited, p, player_id=potential_winner):
                return True
        return False

    def actions_to_index_mapping(self):
        mapping = {}
        i = 0
        for row in self.current_state:
            for cell in row:
                pos = (cell.row, cell.col)
                mapping[pos] = i
                i += 1
        return mapping

    def visualize(self, frame_interval=1):
        if not self.track_history:
            self.history = []
            self.history.append((None, self.current_state))

        # Start by creating a node graph that represents the board's size and neighbor edges,
        # fill node values with first board configuration

        graph = nx.Graph()
        pos_dict = {}

        for row in range(self.size):
            for col in range(self.size):
                cell = self.current_state[row][col]
                graph.add_node(cell)
                pos_dict[cell] = (- cell.row + cell.col, - cell.row - cell.col)

        #  Add edges to graph
        for n in graph.nodes:
            for neighbor in n.neighbors:
                graph.add_edge(n, neighbor)

        # Start visualization process
        plt.switch_backend(newbackend="macosx")
        # plt.show()

        for action, board_state in self.history:
            # Start by creating a node graph that represents the board's size and neighbor edges,
            # fill node values with first board configuration

            graph = nx.Graph()
            pos_dict = {}
            for row in range(self.size):
                for col in range(self.size):
                    cell = board_state[row][col]
                    graph.add_node(cell)
                    pos_dict[cell] = (- cell.row +
                                      cell.col, - cell.row - cell.col)

            #  Add edges to graph
            for n in graph.nodes:
                for neighbor in n.neighbors:
                    graph.add_edge(n, neighbor)

            labels = {}
            clear_labels = {}
            player_one_nodes = []
            player_two_nodes = []
            clear_nodes = []
            fill_lists_dict = {1: player_one_nodes,
                               2: player_two_nodes,
                               0: clear_nodes}
            moved_node = None
            player_color = {1: "blue",
                            2: "red",
                            0: "white"}

            if action is not None:
                moved_node = board_state[action[0]][action[1]]
                pos_dict[moved_node] = (- moved_node.row +
                                        moved_node.col, - moved_node.row - moved_node.col)

            for n in graph.nodes:
                labels[n] = str((n.row, n.col))
                fill_lists_dict[n.filled_by].append(
                    n)  # Add node to correct list

            # Draw all node types
            for pid in fill_lists_dict.keys():
                nx.draw(graph, pos=pos_dict, nodelist=fill_lists_dict[pid],
                        node_color=player_color[pid], edgecolors='black', node_size=1000)
                # Fill in labels (position)
                font_color = "black" if pid == 0 else "white"
                nx.draw_networkx_labels(
                    graph, pos_dict, labels, font_size=11, font_color=font_color)

            # Draw moved node
            if moved_node is not None:
                nx.draw(graph, pos=pos_dict, nodelist=[moved_node],
                        node_color=player_color[moved_node.filled_by], edgecolors='black', node_size=1300)

            plt.pause(frame_interval)
            plt.clf()


class DisjSet:
    def __init__(self, n):
        # Constructor to create and
        # initialize sets of n items
        self.rank = [1] * n
        self.parent = [i for i in range(n)]

    # Finds set of given item x
    def find(self, x):

        # Finds the representative of the set
        # that x is an element of
        if (self.parent[x] != x):
            # if x is not the parent of itself
            # Then x is not the representative of
            # its set,
            self.parent[x] = self.find(self.parent[x])

            # so we recursively call Find on its parent
            # and move i's node directly under the
            # representative of this set

        return self.parent[x]

    # Do union of two sets represented
    # by x and y.
    def Union(self, x, y):

        # Find current sets of x and y
        xset = self.find(x)
        yset = self.find(y)

        # If they are already in same set
        if xset == yset:
            return

        # Put smaller ranked item under
        # bigger ranked item if ranks are
        # different
        if self.rank[xset] < self.rank[yset]:
            self.parent[xset] = yset

        elif self.rank[xset] > self.rank[yset]:
            self.parent[yset] = xset

        # If ranks are same, then move y under
        # x (doesn't matter which one goes where)
        # and increment rank of x's tree
        else:
            self.parent[yset] = xset
            self.rank[xset] = self.rank[xset] + 1

