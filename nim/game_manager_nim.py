import copy
from environment.nim import Nim


class GameManager:
    """
    Produce initial game states
    Generate child states from a given parent state
    Recognizes winning states
    """

    def __init__(self, final_reward, num_rocks, max_draw):
        self.final_reward = final_reward
        self.num_rocks = num_rocks
        self.max_draw = max_draw
        self.real_game = Nim(num_rocks=num_rocks,
                             max_draw=max_draw)

    def reset_real_game(self):
        self.real_game = Nim(num_rocks=self.num_rocks,
                             max_draw=self.max_draw)

    def get_legal_actions_and_corresponding_child_states(self, state, pid):
        """
        Generates actions and their consequent states for a given input state
        :param state: state to perform actions from (just a number in a list lol)
        :returns: list of (action, state_prime)-pairs
        """
        sim_game = Nim(num_rocks=state[0],  # IMPORTANT, WE WANT TO CONTINUE FROM WHERE WE LEFT OFF, NOT NEW GAME
                       max_draw=self.max_draw,
                       player_id=pid)
        child_states = []
        reward = None
        if sim_game.is_finished():
            reward = self.final_reward * -1*pid
        legal_actions = sim_game.get_actions()
        for action in legal_actions:
            child_game = copy.deepcopy(sim_game)
            child_game.perform_action(action)
            child_states.append(child_game.get_state()[0])
            reward = None

        player_id = pid*-1
        return reward, legal_actions, child_states, player_id

    def get_legal_actions(self, num_rocks=None):
        """"
        Works for both real game and for simulations. Input state assumes simulation
        If None given, returns legal actions in real life game
        """
        if type(num_rocks) == list:
            num_rocks = num_rocks[0]
        if num_rocks is not None:
            return Nim(num_rocks=num_rocks,
                       max_draw=self.max_draw).get_actions()
        else:
            return self.real_game.get_actions()
