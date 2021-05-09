from simulator.mcts import MCTS, first_cell_heuristic, apply_heuristics
from environment.game_manager import GameManager
from agent.actor import Actor
from tournament.tournament import TOPP
from datetime import datetime
import yaml
import random
import time
import matplotlib.pyplot as plt

config = yaml.full_load(open("config/config_demo_5x5.yml"))
actor_cfg = config["Actor"]
env_cfg = config["Environment"]
training_cfg = config["Training"]
topp_cfg = config["Topp"]


def show_epsilon(first_epsilon, final_epsilon, num_games):
    decay = final_epsilon ** (1 / num_games)
    x = [i for i in range(num_games)]
    y = [first_epsilon * decay ** i for i in range(num_games)]
    ones = [1] * num_games
    zeros = [0] * num_games
    plt.plot(x, y)
    plt.plot(x, ones)
    plt.plot(x, zeros)
    plt.show()


def main(resume_params=None):
    """
    This function performs a number of episodes, inside each of which a Monte Carlo Tree search provides a distribution
    for actions along the way. The state-action pairs are recorded in a replay buffer with the actor, from which a mini-
    batch is drawn and trained on between each episode.


    NOTE: Some game-specific heuristics are applied along the way. These are the only non-modular components of this
    algorithm.
    """

    # MCTS-specific params
    num_actual_games = training_cfg["number_of_episodes"]
    number_of_search_games = training_cfg["number_of_search_games"]
    final_sg = training_cfg["final_sg"]
    sg_reductions = training_cfg["sg_reductions"]
    search_game_reduction = int(
        (number_of_search_games - final_sg) / sg_reductions)
    mcts_epsilon = training_cfg["mcts_epsilon"]
    mcts_epsilon_decay = training_cfg["mcts_epsilon_decay"]

    # Actor-specific params
    game_manager = GameManager(
        env_cfg)  # Create game_manager to provide action to index mapping for input vector to actor ANET
    mapping = game_manager.get_actions_to_idx_mapping()
    # Want exponential decay towards final
    epsilon_decay = actor_cfg["final_epsilon"] ** (1 / num_actual_games)

    actor = Actor(actor_cfg, mapping=mapping, epsilon_decay=epsilon_decay)
    if resume_params is not None:
        actor.load(parameters=resume_params[0], weights=resume_params[1])

    # TOPP/SAVING/VISUALIZING-specific params
    save_interval = num_actual_games // (topp_cfg["M"] - 1)
    # A list of episode numbers we want to display gameplay for
    show_episodes = training_cfg["show_episodes"]
    m = 0  # For saving models with numbered names

    # Begin simulating "real" episodes
    for g_a in range(num_actual_games):

        if (g_a + 1) % (num_actual_games / sg_reductions) == 0:
            # Reduce search games stepwise to save comp. time
            number_of_search_games -= search_game_reduction

        # For visualizing during training
        record = g_a in show_episodes
        # Reset the game for a new episode
        game_manager.reset_real_game(record=record)

        current_state, player_id = game_manager.get_real_game_state()
        # Generate search agent with root set to current_state (which has just been reset to empty)
        mcts = MCTS(root_state=current_state,
                    player_id=player_id,
                    num_search_games=number_of_search_games,
                    default_policy=actor,
                    game_manager=game_manager)
        if g_a == 0:
            # Save first, untrained model
            actor.save(actor.model, f"{actor_cfg['path_to_trained']}{m}")
            print("saved model:", m)
            m += 1

        i = 0
        while not game_manager.real_game_is_finished():  # Loop over actual actions in real life
            i += 1
            if i == 1:
                action_dist_targets = first_cell_heuristic(
                    current_state, mapping)
            else:
                START_TIME = datetime.now()
                # Perform actual Monte Carlo Tree Search
                temp_action_dist_targets = mcts.search(
                    search_from_state=current_state, player_id=player_id)
                action_dist_targets = apply_heuristics(temp_action_dist_targets,
                                                       current_state,
                                                       player_id,
                                                       game_manager)
                print(f'Runtime of a mcts search of length {mcts.num_search_games}:', datetime.now(
                ) - START_TIME)

            # Action_dist_targets is a dictionary of action->probability pairs
            vectorized_state = current_state.copy()
            vectorized_state.append(player_id)
            actor.add_to_replay((vectorized_state, action_dist_targets))

            # In case we want to introduce some randomness in real games as well
            if random.randint(0, 1000) / 1000 <= mcts_epsilon:
                action_to_perform = max(
                    action_dist_targets.keys(), key=lambda a: action_dist_targets[a])
            else:
                action_to_perform = random.choice(
                    list(action_dist_targets.keys()))

            # Shift real game into new state (update of player id happens here)
            game_manager.perform_real_game_action(action_to_perform)
            current_state, player_id = game_manager.get_real_game_state()

        if record:
            game_manager.show_real_game()  # Display the episode
        actor.decay_epsilon()
        print_round(g_a)
        actor.print_accuracy()  # Print accuracy before training
        actor.train(epochs=actor_cfg["epochs"])
        actor.print_accuracy()  # Print accuracy after training once

        if (g_a + 1) % save_interval == 0:
            actor.save(actor.model, f"{actor_cfg['path_to_trained']}{m}")
            print("saved model:", m)
            m += 1
    actor.visualize_loss()


def print_round(g_a):
    counts = {1: "st", 2: "nd", 3: "rd"}
    suffix = "th"
    if int((g_a + 1) % 10) in counts.keys():
        suffix = counts[int((g_a + 1) % 10)]
    print(f"Training actor for the {g_a + 1}{suffix} time")


def topp(argmax=True):
    tournament = TOPP(topp_config=topp_cfg,
                      actor_config=actor_cfg, game_manager_config=env_cfg)
    tournament.play_tournament(argmax=argmax)


def play_against_agent(go_first=False, agent=None):
    """
    Function to play against input agent or against MCTS tree search itself
    """
    user_pid = 1 if go_first else 2
    game_manager = GameManager(env_cfg)
    current_state, player_id = game_manager.get_real_game_state()
    mcts = None
    if agent is None:
        mcts = MCTS(root_state=current_state,
                    player_id=player_id,
                    num_search_games=training_cfg["number_of_search_games"],
                    default_policy=None,
                    game_manager=game_manager)
    while not game_manager.real_game_is_finished():  # Loop over actual actions in real life
        game_manager.show_real_game()
        legals = game_manager.get_legal_actions()
        if player_id == user_pid:
            legaldict = {}
            for i, a in enumerate(legals):
                legaldict[i] = a
            action_to_perform = legaldict[int(
                input(f"Please provide an index from the list:\n{legaldict}"))]
        else:

            if agent is None:
                action_dist_targets = mcts.search(current_state, player_id)
                mcts.vizualize_search_tree()
                action_to_perform = max(
                    action_dist_targets.keys(), key=lambda a: action_dist_targets[a])
            else:
                vectorized_state = current_state.copy()
                vectorized_state.append(player_id)
                action_to_perform = agent.get_action(vectorized_state, legals)

        game_manager.perform_real_game_action(action_to_perform)
        current_state, player_id = game_manager.get_real_game_state()


def load_actor(gm, params, weights):
    mapping = gm.get_actions_to_idx_mapping()
    return Actor(actor_cfg, mapping=mapping, params=params, weights=weights)


def show_gameplay_between_actors(first_player, second_player, game_manager):
    """
    Takes in two actors and a game manager and displays a game between them
    """
    current_state, player_id = game_manager.real_game.get_state()
    actors = {1: first_player, 2: second_player}
    while not game_manager.real_game_is_finished():  # Loop over actual actions in real life
        legals = game_manager.get_legal_actions()
        vectorized_state = current_state.copy()
        vectorized_state.append(player_id)
        actor = actors[player_id]
        action_to_perform = actor.get_action(vectorized_state, legals)

        game_manager.perform_real_game_action(action_to_perform)
        game_manager.show_real_game()
        time.sleep(0.5)
        current_state, player_id = game_manager.get_real_game_state()


if __name__ == '__main__':
    game_manager = GameManager(env_cfg)

    topp(argmax=True)  # TOPP where we choose best action from agent every the time
    # TOPP where we choose from actions based on distribution from agent
    topp(argmax=False)
    topp(argmax=False)

    actor1 = load_actor(
        gm=game_manager, params=f"{actor_cfg['path_to_trained']}" + "0.json", weights=f"{actor_cfg['path_to_trained']}" + "0.h5")
    actor2 = load_actor(
        gm=game_manager, params=f"{actor_cfg['path_to_trained']}" + "4.json", weights=f"{actor_cfg['path_to_trained']}" + "4.h5")
    show_gameplay_between_actors(actor1, actor2, game_manager)
