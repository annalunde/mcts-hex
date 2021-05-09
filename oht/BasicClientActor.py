import math
import yaml
import numpy as np
from BasicClientActorAbs import BasicClientActorAbs
from agent.actor import Actor
from environment.game_manager import GameManager
from time import sleep


class BasicClientActor(BasicClientActorAbs):

    def __init__(self, IP_address=None, verbose=True):
        self.series_id = -1
        BasicClientActorAbs.__init__(self, IP_address, verbose=verbose)
        self.series_num = -1

        self.boardsize = 6
        self.first_player = 1
        self.game_manager = GameManager(
            None, self.boardsize, self.first_player)

        self.mapping = self.game_manager.real_game.actions_to_index_mapping()

        self.actor = Actor(config=None, mapping=self.mapping,
                       params="models/Actor2/resumed_actor29.json", weights="models/Actor2/resumed_actor29.h5")


    def handle_get_action(self, state):
        """
        Here you will use the neural net that you trained using MCTS to select a move for your actor on the current board.
        Remember to use the correct player_number for YOUR actor! The default action is to select a random empty cell
        on the board. This should be modified.
        :param state: The current board in the form (1 or 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), where
        1 or 2 indicates the number of the current player.  If you are player 2 in the current series, for example,
        then you will see a 2 here throughout the entire series, whereas player 1 will see a 1.
        :return: Your actor's selected action as a tuple (row, column)
        """
        player_id = state[0]
        #print(f"Player id is {player_id}, and id of beginning player was {self.starting_player}")
        current_state = list(state[1:])
        current_state = np.array(current_state).reshape(-1, self.boardsize)
        if self.starting_player == 2:
            # Player 1 is always starting in our logic, and plays horizontally.
            # This means we only have to switch 1s and 2s and play as normal
            player_id = 1 if player_id == 2 else 2
            # Need to switch all 1's to 2's in state and vice versa.
            swapped_state = np.zeros(shape=current_state.shape)
            for i, row in enumerate(current_state):
                for j, val in enumerate(row):
                    if val == 0:
                        swapped_state[i][j] = 0
                    elif val == 1:
                        swapped_state[i][j] = 2
                    elif val == 2:
                        swapped_state[i][j] = 1
            current_state = swapped_state
        else:
            # Then player 1 begins as with our logic, but since 1 plays horizontally in our logic we need to transpose
            # The board, so that the vertical axis becomes horizontal, and vice versa
            current_state = current_state.T
        current_state = current_state.reshape(-1,)
        # Converting back to python list with int
        current_state = [int(val) for val in current_state]

        # Update real game to represent state that is received
        self.game_manager.update_real_game(current_state, player_id)
        # self.game_manager.real_game.visualize()  # Show state of board just after opponent has acted

        legals = self.game_manager.get_legal_actions(current_state)


        vectorized_state = current_state
        vectorized_state.append(player_id)
        action = self.actor.get_action(vectorized_state, legals)

        self.game_manager.real_game.perform_action(action)
        # self.game_manager.real_game.visualize()  # Show current state of board after performing our action

        if self.starting_player == 1:
            next_move = (action[1], action[0])  # Transpose back to original
        else:
            next_move = action
        #print("Your model chose the move {}".format(next_move))

        return next_move

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """
        Set the player_number of our actor, so that we can tell our MCTS which actor we are.
        :param unique_id - integer identifier for the player within the whole tournament database
        :param series_id - (1 or 2) indicating which player this will be for the ENTIRE series
        :param player_map - a list of tuples: (unique-id series-id) for all players in a series
        :param num_games - number of games to be played in the series
        :param game_params - important game parameters.  For Hex = list with one item = board size (e.g. 5)
        :return

        """
        self.series_num += 1  # Increment series number to represent current index in actor list
        self.series_id = series_id

    def handle_game_start(self, start_player):
        """
        :param start_player: The starting player number (1 or 2) for this particular game.
        :return
        """
        self.starting_player = start_player

    def handle_game_over(self, winner, end_state):
        """
        Here you can decide how to handle what happens when a game finishes. The default action is to print the winner and
        the end state.
        :param winner: Winner ID (1 or 2)
        :param end_state: Final state of the board.
        :return:
        """
        # kan visualisere her

        print("Game over,")
        out = "You won" if winner == self.series_id else "You lost"
        print(out)
        # self.game_manager.update_real_game(end_state)  # Update real game to represent end_state that is received
        # self.game_manager.real_game.visualize()  # Show state of board

    def handle_series_over(self, stats):
        """
        Here you can handle the series end in any way you want; the initial handling just prints the stats.
        :param stats: The actor statistics for a series = list of tuples [(unique_id, series_id, wins, losses)...]
        :return:
        """
        print(
            f"Series ended, you were player {self.series_id} and these are the stats:")
        print(str(stats))

    def handle_tournament_over(self, score):
        """
        Here you can decide to do something when a tournament ends. The default action is to print the received score.
        :param score: The actor score for the tournament
        :return:
        """

        print("Tournament over. Your score was: " + str(score))

    def handle_illegal_action(self, state, illegal_action):
        """
        Here you can handle what happens if you get an illegal action message. The default is to print the state and the
        illegal action.
        :param state: The state
        :param action: The illegal action
        :return:
        """

        print("An illegal action was attempted:")
        print('State: ' + str(state))
        print('Action: ' + str(illegal_action))


if __name__ == '__main__':
    bsa = BasicClientActor(verbose=True)
    bsa.connect_to_server()
