""" A toy example of playing against pretrianed AI on Leduc Hold'em
"""
from tqdm import tqdm
from mcts import MCTS, TreeSearch, init_game
from mcts_ev import MCTS_Expected

import rlcard
from rlcard import models
from rlcard.agents import CFRAgent
from rlcard.agents import RandomAgent
from rlcard.utils import print_card
import types
import numpy as np
from overrides import run
from treesearch import TreeSearch


def step(self, state):
    return self.eval_step(state)[0]


trials = 5
parameter_testing = {}
for rollouts_num in [10, 50, 100, 250, 500]:
    rolls = []
    for trial in range(trials):
        env = rlcard.make("leduc-holdem")
        env.game.init_game = types.MethodType(init_game, env.game)

        cfr_agent = CFRAgent(env)
        random_agent = RandomAgent(num_actions=env.num_actions)
        mcts_agent = MCTS_Expected(env, rollouts_num, 0)
        rule2_agent = models.load("leduc-holdem-rule-v1").agents[0]
        cfr_agent.step = types.MethodType(step, cfr_agent)
        env.set_agents([mcts_agent, random_agent])

        rewards_vs_random = []
        for i in tqdm(range(250)):
            trajectories, payoffs = env.run(is_training=True)
            rewards_vs_random.append(payoffs[0])
        final_reward = sum(rewards_vs_random)
        rolls.append(final_reward)

    parameter_testing[rollouts_num] = np.mean(rolls)

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()
plt.plot(range(len(list(parameter_testing.keys()))), list(parameter_testing.values()))
plt.xlabel("Number of Rollouts")
plt.ylabel("Final Reward")
plt.xticks(
    ticks=range(len(list(parameter_testing.keys()))),
    labels=list(parameter_testing.keys()),
)
plt.title("Hyperparameter Tuning by Number of Rollouts")
plt.savefig("pics/parameters.png")