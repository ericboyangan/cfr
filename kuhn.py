import json
import os
import numpy
import random

def choose(choices, dist):
    norm = sum([d for d in dist if d > 0])
    if norm > 0:
        choice = numpy.random.choice(range(len(choices)), p=[d * 1.0 / norm if d > 0 else 0 for d in dist])
    else:
        choice = numpy.random.choice(range(len(choices)))
    return choices[choice]

class Kuhn:
    def __init__(self, player_0, player_1, p0_name="player 0", p1_name="player 1"):
        player_0.reset()
        player_1.reset()
        self.players = [player_0, player_1]
        self.names = [p0_name, p1_name]
        self.cards = random.choice(["01", "02", "10", "12", "20", "21"])
        self.game_state = self.cards

    def playthrough(self, print_result=False):
        while True:
            payoff = Kuhn.payoff(self.game_state)
            if payoff is not None:
                if print_result:
                    print("Game ended: {} wins {}. Replay: {}".format(
                        self.names[0] if payoff > 0 else self.names[1], abs(payoff), self.game_state))
                return payoff
            active_player = len(self.game_state) % 2
            info_set = self.game_state[active_player] + self.game_state[2:]
            action = self.players[active_player].act(info_set)
            self.game_state += action
    
    @staticmethod
    def payoff(game_state):
        if len(game_state) <= 3:
            return None
        if game_state[-2:] == "pp":
            return 1 if game_state[0] > game_state[1] else -1
        if game_state[-2:] == "bb":
            return 2 if game_state[0] > game_state[1] else -2
        if game_state[-2:] == "bp":
            return 1 if len(game_state) % 2 == 0 else -1

class HumanKuhner:
    def __init__(self):
        pass
    
    def reset(self):
        pass

    def act(self, info_set):
        while True:
            display = "You hold {}. Action so far: {}. Pass/bet (p/b)?".format(info_set[0], info_set[1:])
            choice = raw_input(display)
            if choice in ("p", "b"):
                break
        return choice

class RandomKuhner:
    def __init__(self):
        pass
    
    def reset(self):
        pass

    def act(self, info_set):
        return random.choice(["p", "b"])

class TightAggressiveKuhner:
    def __init__(self):
        pass

    def reset(self):
        pass

    def act(self, info_set):
        if info_set[0] == "0":
            return "p"
        if info_set[0] == "2":
            return "b"
        return random.choice(["p", "b"])

class OneShotCFRKuhner:
    def __init__(self, autoload=False):
        self.p0_strategies = []
        self.p0_regrets = []
        self.p0_cumulative_prob = []
        p0_metas = ["check_then_fold", "check_then_raise", "bet_out"]
        for meta_0 in p0_metas:
            for meta_1 in p0_metas:
                for meta_2 in p0_metas:
                    self.p0_strategies.append((meta_0, meta_1, meta_2))
                    self.p0_regrets.append(0)
                    self.p0_cumulative_prob.append(0)
        self.p1_strategies = []
        self.p1_regrets = []
        self.p1_cumulative_prob = []
        p1_metas = ["check_or_fold", "bet_or_fold", "check_or_raise", "bet_or_raise"]
        for meta_0 in p1_metas:
            for meta_1 in p1_metas:
                for meta_2 in p1_metas:
                    self.p1_strategies.append((meta_0, meta_1, meta_2))
                    self.p1_regrets.append(0)
                    self.p1_cumulative_prob.append(0)
        self.selected_p0_strategy = None
        self.filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oneshot.json")
        if autoload:
            self.load()


    def calculate_payoff(self, cards, p0_strategy, p1_strategy):
        p0_meta = p0_strategy[cards[0]]
        p1_meta = p1_strategy[cards[1]]
        if p0_meta == "bet_out":
            if p1_meta[-4:] == "fold":
                return 1
            else:
                return 2 if cards[0] > cards[1] else -2
        else:
            if p1_meta[:5] == "check":
                return 1 if cards[0] > cards[1] else -1
            else:
                if p0_meta == "check_then_fold":
                    return -1
                else:
                    return 2 if cards[0] > cards[1] else -2

    def update_cumulative_prob(self):
        norm = sum([r for r in self.p0_regrets if r > 0])
        if norm > 0:
            for i in range(len(self.p0_strategies)):
                self.p0_cumulative_prob[i] += (1.0 * self.p0_regrets[i] / norm if self.p0_regrets[i] > 0 else 0)
        else:
            for i in range(len(self.p0_strategies)):
                self.p0_cumulative_prob[i] += (1.0 / len(self.p0_strategies))
        norm = sum([r for r in self.p1_regrets if r > 0])
        if norm > 0:
            for i in range(len(self.p1_strategies)):
                self.p1_cumulative_prob[i] += (1.0 * self.p1_regrets[i] / norm if self.p1_regrets[i] > 0 else 0)
        else:
            for i in range(len(self.p1_strategies)):
                self.p1_cumulative_prob[i] += (1.0 / len(self.p1_strategies))

    def train(self, iters=1):
        for i in range(iters):
            p0_strategy = choose(self.p0_strategies, self.p0_regrets)
            p1_strategy = choose(self.p1_strategies, self.p1_regrets)
            self.update_cumulative_prob()
            cards = [0, 1, 2]
            random.shuffle(cards)
            actual_payoff = self.calculate_payoff(cards, p0_strategy, p1_strategy)
            for i, cf_strategy in enumerate(self.p0_strategies):
                cf_payoff = self.calculate_payoff(cards, cf_strategy, p1_strategy)
                self.p0_regrets[i] += (cf_payoff - actual_payoff)
            for i, cf_strategy in enumerate(self.p1_strategies):
                cf_payoff = self.calculate_payoff(cards, p0_strategy, cf_strategy)
                self.p1_regrets[i] += (actual_payoff - cf_payoff)
    
    def reset(self):
        self.selected_p0_strategy = None

    def act(self, info_set):        
        card = int(info_set[0])
        history = info_set[1:]
        active_player = len(history) % 2
        if active_player == 1:
            p1_strategy = choose(self.p1_strategies, self.p1_regrets)
            meta = p1_strategy[card]
            if history[-1] == "p":
                if meta[:5] == "check":
                    return "p"
                return "b"
            if meta[-4:] == "fold":
                return "p"
            return "b"
        if self.selected_p0_strategy is None:
            self.selected_p0_strategy = choose(self.p0_strategies, self.p0_regrets)
        meta = self.selected_p0_strategy[card]
        if history == "":
            if meta == "bet_out":
                return "b"
            return "p"
        if meta == "check_then_fold":
            return "p"
        return "b"

    def save(self):
        with open(self.filename, "w") as f:
            json.dump({
                "p0_strategies": self.p0_strategies,
                "p0_regrets": self.p0_regrets,
                "p0_cumulative_prob": self.p0_cumulative_prob,
                "p1_strategies": self.p1_strategies,
                "p1_regrets": self.p1_regrets,
                "p1_cumulative_prob": self.p1_cumulative_prob,
            }, f)

    def load(self):
        with open(self.filename, "r") as f:
            data = json.load(f)
            self.p0_strategies = data["p0_strategies"]
            self.p0_regrets = data["p0_regrets"]
            self.p0_cumulative_prob = data["p0_cumulative_prob"]
            self.p1_strategies = data["p1_strategies"]
            self.p1_regrets = data["p1_regrets"]
            self.p1_cumulative_prob = data["p1_cumulative_prob"]

    def print_strategy(self):
        norm = sum(self.p0_cumulative_prob)
        print("One Shot CFR Kuhner ({} iters)".format(int(norm)))
        print("Player 0 strategy distribution:")
        for s, p in zip(self.p0_strategies, self.p0_cumulative_prob):
            print("On 0: {:20}On 1: {:20}On 2: {:20}{:4.2f}%".format(s[0], s[1], s[2], p * 100 / norm))
        print("Player 1 strategy distribution:")
        for s, p in zip(self.p1_strategies, self.p1_cumulative_prob):
            print("On 0: {:20}On 1: {:20}On 2: {:20}{:4.2f}%".format(s[0], s[1], s[2], p * 100 / norm))

class CFRKuhner:
    ACTIONS = ["p", "b"]

    def __init__(self, autoload=False):
        self.node_map = {}
        self.filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfr.json")
        if autoload:
            self.load()

    def save(self):
        with open(self.filename, "w") as f:
            json.dump(self.node_map, f)

    def load(self):
        with open(self.filename, "r") as f:
            self.node_map = json.load(f)

    def reset(self):
        pass

    def get_regret_matched_strategy(self, info_set):
        regrets = self.node_map[info_set]["regrets"]
        norm = sum([r for r in regrets if r > 0])
        if norm == 0:
            return [1.0 / len(CFRKuhner.ACTIONS) for a in CFRKuhner.ACTIONS]
        else:
            return [r / norm if r > 0 else 0 for r in regrets]

    def act(self, info_set):
        return choose(CFRKuhner.ACTIONS, self.node_map[info_set]["cumulative_prob"])
    
    def update_cumulative_prob(self, info_set, updates):
        for i, p in enumerate(updates):
            self.node_map[info_set]["cumulative_prob"][i] += p

    def compute_cfr_value(self, cards, history, p0, p1):
        """
        Recursively computes game value.
          - `cards` and `history` describe the current game state.
          - `p0` and `p1` are the counterfactual reach probabilities of the current node, from
            player 0 and 1's perspective respectively.
        """
        active_player = len(history) % 2
        game_state = "{}{}{}".format(cards[0], cards[1], history)
        payoff = Kuhn.payoff(game_state)
        # Return payoff if terminal.
        if payoff is not None:
            return payoff * (1 if active_player == 0 else -1)
        info_set = "{}{}".format(cards[active_player], history)
        if info_set not in self.node_map:
            self.node_map[info_set] = {
                "regrets": [0 for a in CFRKuhner.ACTIONS],
                "cumulative_prob": [0.0 for a in CFRKuhner.ACTIONS]
            }
        # Use basic regret-matching to obtain the strategy employed in the current node.
        current_strategy = self.get_regret_matched_strategy(info_set)
        # Accumulate the current node's strategy, weighted by the probability this node is reached.
        realization_probability = p0 if active_player == 0 else p1
        current_strategy_update_probabilities = [realization_probability * p for p in current_strategy]
        self.update_cumulative_prob(info_set, current_strategy_update_probabilities)
        # Contains the utility of taking each possible action at this point.
        action_utilities = []
        # Accumulates the expected utility of the current node, employing the current node's strategy.
        current_node_utility = 0
        for i, action in enumerate(CFRKuhner.ACTIONS):
            next_history = history + action
            # Recurse. The current node's utility after taking an action is the inverse of the utility of the
            # node that follows (Kuhn is zero-sum.) If player `i` takes the action...
            if active_player == 0:
                next_utility = -1.0 * self.compute_cfr_value(cards, next_history, p0 * current_strategy[i], p1)
            else:
                next_utility = -1.0 * self.compute_cfr_value(cards, next_history, p0, p1 * current_strategy[i])
            action_utilities.append(next_utility)
            current_node_utility += next_utility * current_strategy[i]
        # Accumulate counterfactual regrets.
        for i, action in enumerate(CFRKuhner.ACTIONS):
            regret = action_utilities[i] - current_node_utility
            self.node_map[info_set]["regrets"][i] += (p1 if active_player == 0 else p0) * regret
        # Finally, the expected utility of the current node is returned.
        return current_node_utility

    def train(self, iters=1):
        total_util = 0
        for i in range(iters):
            cards = [0, 1, 2]
            random.shuffle(cards)
            total_util += self.compute_cfr_value(cards, "", 1.0, 1.0)
        print("Training complete ({} iters). Average game value = {}".format(iters, total_util / iters))

    def print_strategy(self):
        for node in sorted(self.node_map.keys(), key=lambda info_set: (len(info_set), info_set)):
            strategy = self.node_map[node]["cumulative_prob"]
            norm = sum(strategy)
            print("{:5} pass: {:.3f} bet: {:.3f}".format(node, strategy[0] / norm, strategy[1] / norm))

if __name__ == "__main__":
    # N = 1000000
    # kuhners = [RandomKuhner(), TightAggressiveKuhner(), OneShotCFRKuhner(autoload=True), CFRKuhner(autoload=True)]
    # for i in range(len(kuhners)):
    #     payoffs = []
    #     for j in range(len(kuhners)):
    #         if j >= i:
    #             payoff = 0
    #             for n in range(N):
    #                 game = Kuhn(kuhners[i], kuhners[j]) if n % 2 == 0 else Kuhn(kuhners[j], kuhners[i])
    #                 payoff += game.playthrough() * (1 if n % 2 == 0 else -1)
    #             payoffs.append(1.0 * payoff / N)
    #         else:
    #             payoffs.append(None)
    #     print("\t".join(["{:.3f}".format(p) if p is not None else " "*4 for p in payoffs]))
    # 0.001   -0.207  -0.098  -0.130
    #         0.002   -0.108  -0.039
    #                 0.001   -0.002
    #                         -0.001
    score = 0
    games = 0
    name = raw_input("Enter your name: ")
    while True:
        game = (Kuhn(HumanKuhner(), CFRKuhner(autoload=True), name, "cfrbot") if games % 2 == 0 else
                Kuhn(CFRKuhner(autoload=True), HumanKuhner(), "cfrbot", name))
        score += game.playthrough(print_result=True) * (1 if games % 2 == 0 else -1)
        games += 1
        print("Score: {}. Games played: {}".format(score, games))