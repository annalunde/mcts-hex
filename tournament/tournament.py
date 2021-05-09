from agent.actor import Actor
from environment.game_manager import GameManager
import random
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm


class TOPP:

    def __init__(self, topp_config, actor_config, game_manager_config):
        # number of games to play a round-robin tournament
        self.G = topp_config["G"]
        self.M = topp_config["M"]  # number of actors saved
        self.path_to_agents = actor_config["path_to_trained"]
        self.game_manager = GameManager(game_manager_config)
        self.agents = self.load_agents()
        self.show_players = topp_config["show_players"]

    def load_agents(self):
        agents = []
        mapping = self.game_manager.get_actions_to_idx_mapping()
        # for m in range(self.M):
        for m in range(self.M):
            actor = Actor(config=None, mapping=mapping,
                          params=f"{self.path_to_agents}{str(m)}.json", weights=f"{self.path_to_agents}{str(m)}.h5")
            agents.append(actor)
        return agents

    def play_tournament(self, argmax):
        # Playing the tournament with different actors
        agents = self.load_agents()
        labeled_agents = [(i, agent) for i, agent in enumerate(agents)]
        statistics = {}
        for m in range(self.M):
            statistics[m] = 0
        labeled_agent_pairs = itertools.combinations(
            labeled_agents, 2)  # Get all combinations of players
        for pair in tqdm(labeled_agent_pairs, colour='#39ff14'):
            agent_id1, agent1 = pair[0][0], pair[0][1]
            agent_id2, agent2 = pair[1][0], pair[1][1]

            for i in range(self.G):
                actor1_goes_first = i >= self.G/2
                show = False

                if (agent_id1, agent_id2) in self.show_players and actor1_goes_first:
                    self.show_players.remove((agent_id1, agent_id2))
                    print(f"Displaying Tournament game between actor {agent_id1} and actor {agent_id2}\n"
                          f"(Actor {agent_id1} goes first)")
                    show = True

                elif (agent_id2, agent_id1) in self.show_players and not actor1_goes_first:
                    self.show_players.remove((agent_id2, agent_id1))
                    print(f"Displaying Tournament game between actor {agent_id2} and actor {agent_id1}\n"
                          f"(Actor {agent_id2} goes first)")
                    show = True
                winner = self.play_single_game(
                    agent1, agent2, argmax, actor1_goes_first, show)
                w_stat = agent_id1 if winner == agent1 else agent_id2
                statistics[w_stat] += 1

        self.visualize_statistics(statistics)
        return statistics

    def play_single_game(self, actor1, actor2, argmax, actor1_goes_first, show):
        # two players playing a game
        if actor1_goes_first:
            actors = {1: actor1, 2: actor2}
        else:
            actors = {2: actor1, 1: actor2}

        self.game_manager.reset_real_game(record=show)

        current_state, player_id = self.game_manager.get_real_game_state()

        while not self.game_manager.real_game_is_finished():  # Loop over actual actions in real life
            legals = self.game_manager.get_legal_actions(current_state)
            vectorized_state = current_state.copy()
            vectorized_state.append(player_id)
            actor = actors[player_id]
            action_to_perform = actor.get_action(
                vectorized_state, legals, argmax=argmax)

            self.game_manager.perform_real_game_action(action_to_perform)
            current_state, player_id = self.game_manager.get_real_game_state()
        if show:
            self.game_manager.show_real_game()
        return actors[1] if player_id == 2 else actors[2]

    def visualize_statistics(self, statistics):
        plt.style.use('ggplot')

        x = [i for i in range(self.M)]
        y = [statistics[i] for i in range(self.M)]
        plt.title("Topp Statistics")
        plt.xlabel("Actor")
        plt.ylabel("Wins")
        plt.bar(x, y, color='blue')
        plt.show()
