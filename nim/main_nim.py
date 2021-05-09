from copy import deepcopy
from tree_node import TreeNode
from mcts import MCTS
from environment.nim import Nim
from environment.game_manager_nim import GameManager
import random
import copy


FINAL_REWARD = 1
NUM_ROCKS = 6
MAX_DRAW = 3
NUM_ACTUAL_GAMES = 1
NUM_SEARCH_GAMES = 50


def main():
    actor = None
    for g_a in range(NUM_ACTUAL_GAMES):
        game_manager = GameManager(final_reward=FINAL_REWARD,
                                   num_rocks=NUM_ROCKS,
                                   max_draw=MAX_DRAW)

        current_state, player_id = game_manager.real_game.get_state()
        # Generate search agent with root set to current_state
        mcts = MCTS(root_state=current_state,
                    player_id=player_id,
                    num_search_games=NUM_SEARCH_GAMES,
                    default_policy=actor,
                    game_manager=game_manager)
        # Collect state - actiondistribution pairs along the way (state/targets)
        replay_buffer = []
        i = 0
        while not game_manager.real_game.is_finished():  # Loop over actual actions in real life
            i += 1
            action_dist_targets = mcts.search(
                search_from_state=current_state, player_id=player_id, epsilon=None)
            # Action_dist_targets is a dictionary of action->probability pairs
            if i == 1:
                mcts.vizualize_search_tree()
            vectorized_state = current_state.append(player_id)
            replay_buffer.append((vectorized_state, action_dist_targets))
            action_to_perform = max(
                action_dist_targets.keys(), key=lambda a: action_dist_targets[a])
            game_manager.real_game.perform_action(action_to_perform)
            current_state, player_id = game_manager.real_game.get_state()

        # actor.train(replay_buffer)
        print(f"Training actor for the {g_a+1}th time")


# main()

if __name__ == '__main__':
    game_manager = GameManager(final_reward=FINAL_REWARD,
                               num_rocks=NUM_ROCKS,
                               max_draw=MAX_DRAW)

    current_state, player_id = game_manager.real_game.get_state()
    mcts = MCTS(root_state=current_state,
                player_id=player_id,
                num_search_games=NUM_SEARCH_GAMES,
                default_policy=None,
                game_manager=game_manager)

    while not game_manager.real_game.is_finished():  # Loop over actual actions in real life
        print(f"Number of rocks left: {current_state}")
        if player_id == 1:
            legals = [a for a in game_manager.get_legal_actions(current_state)]

            action_to_perform = int(
                input(f"Please provide a number from the list:\n{legals}"))
        else:
            action_dist_targets = mcts.search(
                search_from_state=current_state, player_id=player_id, epsilon=None)

            if current_state[0] == 5:
                mcts.vizualize_search_tree()

            # Action_dist_targets is a dictionary of action->probability pairs
            vectorized_state = current_state.append(player_id)
            action_to_perform = max(
                action_dist_targets.keys(), key=lambda a: action_dist_targets[a])
        game_manager.real_game.perform_action(action_to_perform)
        current_state, player_id = game_manager.real_game.get_state()
