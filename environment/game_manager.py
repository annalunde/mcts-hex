from environment.hexboard import Hexboard
from time import sleep
from copy import deepcopy


class GameManager:
    """
    Produce initial game states
    Generate child states from a given parent state
    Recognizes winning states
    """

    def __init__(self, config, boardsize=None, first_player=1):
        if config:
            self.final_reward = config["final_reward"]
            self.boardsize = config["boardsize"]
        else:
            self.final_reward = 1
            self.boardsize = boardsize
        self.real_game = Hexboard(size=self.boardsize, first_player=first_player)
        self.rollout_board = None

    def reset_real_game(self, first_player=1, record=False):
        self.real_game = Hexboard(size=self.boardsize, first_player=first_player, track_history=record)

    def update_real_game(self, state, player_id):
        """
        Updates real game based on the state that is given as input. Used primarily for visualizing
        during oht
        """
        self.real_game = Hexboard(size=self.boardsize, first_player=player_id, init_state=state)

    def perform_rollout_action(self, action):
        return self.rollout_board.perform_action(action)

    def perform_real_game_action(self, action):
        return self.real_game.perform_action(action)

    def get_real_game_state(self):
        return self.real_game.get_state()

    def real_game_is_finished(self):
        return self.real_game.is_finished()

    def show_real_game(self):
        return self.real_game.visualize()

    def get_actions_to_idx_mapping(self):
        return self.real_game.actions_to_index_mapping()

    def rollout_feedback(self):
        reward = None
        vectorized_state, pid = self.rollout_board.get_state()
        vectorized_state.append(pid)
        if self.rollout_board.is_finished():
            # Player who checks legal actions and has none must have lost at this point
            reward = -1 * self.final_reward if self.rollout_board.current_player_id == 1 else self.final_reward
            return reward, vectorized_state, [], None
        return reward, vectorized_state, self.rollout_board.get_actions(), self.rollout_board.current_player_id

    def get_legal_actions_and_corresponding_child_states(self, parent_state, pid):
        """
        THIS IS ONLY USED FOR EXPANSION
        Generates actions and their consequent states for a given input state
        :param parent_state: state to perform actions from
        :returns: reward, legal actions, successor states, (next) player id
        """
        self.rollout_board = Hexboard(size=self.boardsize, first_player=pid, init_state=parent_state)
        if self.rollout_board.is_finished():
            # Player who checks legal actions and has none must have lost at this point
            reward = -1 * self.final_reward if pid == 1 else self.final_reward
            return reward, [], [], None
        actions, successor_states, pid = self.rollout_board.generate_successor_states()
        return None, actions, successor_states, pid

    def get_legal_actions(self, state=None):
        """"
        Works for both real game and for simulations. Input state assumes simulation
        If None given, returns legal actions in real life game
        """
        if state is not None:
            return Hexboard(size=self.boardsize, first_player=None, init_state=state).get_actions()
        else:
            return self.real_game.get_actions()
