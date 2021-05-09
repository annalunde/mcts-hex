# d(s,a,i): dict: if branch (s,a) was traversed on the i'th rollout
# Q(s,a): dict: update after fex 100 rollouts: E_t/N(s,a)
# Number of simulations per move, should be taken in as a parameter
# epsilon, should be taken in as a parameter
from collections import defaultdict
from simulator.tree_node import TreeNode
import random
from math import sqrt, log
from graphviz import Digraph
from copy import deepcopy
from datetime import datetime

prevseen = set()


class MCTS:

    def __init__(self, root_state, player_id, num_search_games, default_policy, game_manager):
        self.tree_policy_dict = defaultdict(lambda: 0)  # Q(s,a)
        self.traversed_rollout = {}  # d(s,a,i)=yes/no
        self.c = 1
        self.state_to_node_map = {}
        self.root = TreeNode(root_state, player_id)
        self.num_search_games = num_search_games
        self.default_policy = default_policy
        self.game_manager = game_manager
        self.boardsize = int(sqrt(len(root_state)))


    def search(self, search_from_state, player_id):

        # Update root to be the head of the relevant subtree if it exists
        # If we have a subtree, move pointer down a level to the relevant subtree

        if self.root.state != search_from_state:
            # Assuming all possible child states have been generated from root
            if str(search_from_state) in self.state_to_node_map.keys():
                new_root = self.state_to_node_map[str(search_from_state)]
            else:
                # If we haven't generated this child (e.g. for games where generating all children is infeasible)
                new_root = TreeNode(state=search_from_state, player_id=player_id)
            self.root = new_root
            self.root.parent_nodes = None

        # If not, subtree does not exist and we are at beginning of a real game, keep root as is.
        tree_node = self.root  # Set root as first node to search from

        for _ in range(self.num_search_games):
            path_w_actions_used = []  # Keeping track of path through tree in order to perform backprop later

            # Use tree policy to search from root to a leaf of mct
            while not tree_node.is_leaf():
                legals = tree_node.get_actions()
                action = self.get_action(tree_node, legal_actions=legals)  # This is the TREE policy choosing action
                path_w_actions_used.append((tree_node, action))
                tree_node = tree_node.traverse(action)

            # Leaf is found, potential expansion happens here,
            # but if reward is not None, expansion is impossible (we're already at terminal)
            # PUT SIMPLY: REWARD IS FOR CURRENT STATE, NONE IF STATE IS NOT FINISHED
            reward, actions, states, player_id = self.game_manager.get_legal_actions_and_corresponding_child_states(
                tree_node.state, tree_node.player_id)  # Rollout board is created here

            if reward is None:  # If not, then we are at bottom in search tree, and no rollout is possible

                # We check for previously seen states, and create new nodes only for those states we have not seen
                unseen_states_and_action_pairs = []
                existing_node_action_pairs = []
                for i, state in enumerate(states):
                    action = actions[i]
                    if str(state) in self.state_to_node_map.keys():
                        node = self.state_to_node_map[str(state)]
                        existing_node_action_pairs.append((node, action))
                    else:
                        unseen_states_and_action_pairs.append((state, action))
                # New nodes are created inside tree_node.set_children(), existing are simply pointed to with action
                new_nodes = tree_node.set_children(existing_node_action_pairs, unseen_states_and_action_pairs,
                                                   player_id)
                for node in new_nodes:  # Update our mapping over seen states
                    self.state_to_node_map[str(node.state)] = node

                action = random.choice(actions)  # Node choice after expansion is random
                path_w_actions_used.append((tree_node, action))
                tree_node = tree_node.traverse(action)  # Could be terminal state
                self.game_manager.perform_rollout_action(action)  # Keep rollout board up to date

                # THIS IS THE ROLLOUT PART
                # Use ANET to choose rollout actions from newly expanded to L to final state (F)
                # Or use random with prob. epsilon
                reward, vectorized_state, actions, player_id = self.game_manager.rollout_feedback()

                # Continue making alternating moves until a reward appears
                while reward is None:

                    if self.default_policy is None:  # So we can play against mcts with a random default policy
                        action = random.choice(actions)
                    else:
                        action = self.default_policy.get_action(vectorized_state, actions, epsilon_greedy=True)

                    self.game_manager.rollout_board.perform_action(action)

                    reward, vectorized_state, actions, player_id = self.game_manager.rollout_feedback()

            else:  # Leaf node is terminal state, backpropagate reward for current state up the tree
                # We keep our old reward
                pass
            tree_node = self.backpropagate(tree_node, reward, path_w_actions_used)

        return tree_node.get_distribution()

    def get_action(self, node, legal_actions):
        """
        Returns action recommended by current policy, with the
        exception of random exploration epsilon percent of the time
        :param legal_actions: list[list[tuple(int,int)]]
        """
        game_state = node.state
        pid = node.player_id
        if pid == 1:
            return max(legal_actions,
                       key=lambda action: self.get_policy(game_state, action) + self.get_exploration_bonus(node,
                                                                                                           action))
        return min(legal_actions,
                   key=lambda action: self.get_policy(game_state, action) - self.get_exploration_bonus(node, action))

    def get_policy(self, state, action):
        """
        !QSA values!
        Returns the policy for a given SAP pair.

        :param state: list[tuple(int,int)]
        :param action: list[tuple(int,int)]
        """

        if (str(state), action) in self.tree_policy_dict.keys():
            return self.tree_policy_dict[(str(state), action)]
        else:
            return 0

    def get_exploration_bonus(self, node, action):
        """
        Returns the exploration bonus for a given SAP pair.

        :param board: list[tuple(int,int)]
        :param action: list[tuple(int,int)]
        """
        n_s = node.number_state
        n_s_a = node.number_state_action[action]
        return self.c * sqrt(log(n_s) / (1 + n_s_a))

    def backpropagate(self, node, reward, path_w_actions_used):
        """
        Executes the backpropagation of the reward along the path, and returns the root of the tree
        For each node that gets updated along the path (i.e. number of visits and total eval) all parents' qsa
        are updated to represent the new value of ending up in the given node.

        :param node: Leaf node with a corresponding reward
        :param reward: int
        :param path_w_actions_used: list of node - action_performed tuples
        """
        node.inc_number_state()
        node.update_total_eval(reward)
        self.update_parent_qsas(node)

        for node, action in reversed(path_w_actions_used):
            node.inc_number_state()
            node.inc_number_state_action(action)
            node.update_total_eval(reward)
            # Perform update for all parents with actions leading into the node (since node has been updated)
            self.update_parent_qsas(node)
        first_node = path_w_actions_used[0][0]
        return first_node

    def update_parent_qsas(self, node):
        """
        Updates Q(s,a) values from parent states s into state s'
        (to be used once value of s' has been updated)

        :param node: s'
        """
        if node.parent_nodes is not None:  # Handle root
            total_eval = node.total_eval
            n_visits = node.number_state
            qsa_val = total_eval / n_visits
            for action, parent in node.parent_nodes.items():
                self.tree_policy_dict[(str(parent.state), action)] = qsa_val

    def vizualize_search_tree(self):
        dot = Digraph()
        printable_label = f"State:{self.root.state}" \
                          f"\nTotEval{self.root.total_eval}" \
                          f"\n{self.root.number_state_action.items()}" \
                          f"\nPlayer ID: {self.root.player_id}"
        dot.node(name=str(self.root.__hash__(
        )), label=printable_label)
        self.add_subtree(dot, self.root)

        dot.render(filename=None, cleanup=True, view=True)

    def add_subtree(self, dot, root):
        global prevseen
        root_id = str(root.__hash__())
        for action, childnode in root.child_nodes.items():
            if (action, childnode) in prevseen:
                continue
            prevseen.add((action, childnode))
            child_id = str(childnode.__hash__())
            qsa = self.tree_policy_dict[(str(root.state), action)]
            state_printable = ""
            if self.boardsize <= 2:  # Just for nim
                state_printable = childnode.state
            else:
                for i in range(self.boardsize):
                    state_printable += str(
                        childnode.state[i * self.boardsize:(i + 1) * self.boardsize])
                    if i != self.boardsize - 1:
                        state_printable += "\n"
            dot.node(name=child_id, label=f"State: \n{state_printable}"
                                          f"\nN(s): {childnode.number_state}"
                                          f"\ntotEval: {childnode.total_eval}"
                                          f"\n{childnode.number_state_action.items()}"
                                          f"\nPlayer ID: {childnode.player_id}")
            edgelabel = f"{action}\nQ(s,a): {qsa}"
            if root.player_id == 1:
                edgelabel += f"\nQsa-Expl: {self.get_policy(root.state, action) + self.get_exploration_bonus(root, action)}"
            else:
                edgelabel += f"\nQsa-Expl: {self.get_policy(root.state, action) - self.get_exploration_bonus(root, action)}"

            dot.edge(root_id, child_id, label=edgelabel)
            if not childnode.is_leaf():
                self.add_subtree(dot, childnode)


# Heuristics for training
def first_cell_heuristic(current_state, mapping):
    """
    Game specific heuristic for overriding first state-target pair because mcts
    might not find the optimal: always put in the middle/on diagonal if the board is empty.
    """
    action_dist_targets = {}
    for action in mapping.keys():
        action_dist_targets[action] = 0
    if sqrt(len(current_state)) % 2 == 0:
        ind = int(len(current_state) // 2 -
                  sqrt(len(current_state)) // 2)  # Ninja mathematics for finding cell idx on diagonal
        k = list(action_dist_targets.keys())[ind]
        action_dist_targets[k] = 1.0
    else:
        ind = len(current_state) // 2
        k = list(action_dist_targets.keys())[ind]
        action_dist_targets[k] = 1.0
    return action_dist_targets


def apply_heuristics(dist, state, pid, game_man):
    """
    If mcts returns an action with more than 50% for a single action, check if the action will immediately lead to a win
    and if so, change the distribution to say 100% for the winning action, and 0 for the other actions.
    """
    check_pid = 2 if pid == 1 else 1  # We want to check previous player (only one who could have won at current state)
    for p, action in enumerate(dist):
        if p > 0.5:
            test_state = state.copy()
            ind = int(action[0] * sqrt(len(state))) + action[1]
            test_state[ind] = pid
            reward, _, _, _ = game_man.get_legal_actions_and_corresponding_child_states(test_state, check_pid)
            if reward is not None:
                dist = dict.fromkeys(dist, 0.)
                dist[action] = 1.0
                return dist
    return dist
