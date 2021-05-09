from collections import defaultdict
import random


class TreeNode:

    def __init__(self, state, player_id, parent_nodes=None):
        self.state = state
        self.player_id = player_id
        self.number_state = 0  # N(s): dict: number of times been in state s
        # N(s,a): dict with actions as keys and counts as values, number of times took action a from state s
        self.number_state_action = defaultdict(lambda: 0)
        # {a1:c1, a2:c2}, action is NOT string, while child is a new TreeNode object
        self.child_nodes = {}
        self.parent_nodes = parent_nodes
        self.total_eval = 0  # total eval for this node t, E_t = E_t + eval

    def set_parent(self, tree_node):
        self.parent = tree_node

    def set_children(self, existing, unseen_states, player_id):
        """
        exisiting contains (node,action) while unseen_states are (state, action) pairs
        """
        new_nodes = []
        for state, action in unseen_states:
            # Create new nodes with single parent in parentlist, and set as my children
            new = TreeNode(state=state, player_id=player_id,
                           parent_nodes={action: self})
            new_nodes.append(new)
            self.child_nodes[action] = new
        for existing_node, action in existing:
            # Add self to parentlist and set as my children
            self.child_nodes[action] = existing_node
            existing_node.parent_nodes[action] = self
        return new_nodes

    def get_parent(self):
        return self.parent

    def get_child_nodes(self):
        return self.child_nodes.values()

    def get_actions(self):
        return self.child_nodes.keys()

    def inc_number_state(self):
        self.number_state += 1

    def inc_number_state_action(self, action):
        self.number_state_action[action] += 1

    def is_leaf(self):
        return len(self.child_nodes.keys()) == 0

    def is_root(self):
        return self.parent_nodes is None

    def update_total_eval(self, evaluation):
        self.total_eval += evaluation

    def traverse(self, action):
        return self.child_nodes[action]

    def get_distribution(self):
        tot_counts = sum(self.number_state_action.values())
        action_prob_dist = {}
        for action in self.number_state_action.keys():
            # Normalize here
            action_prob_dist[action] = self.number_state_action[action]/tot_counts
        return action_prob_dist
