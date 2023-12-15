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
from run import run


def step(self, state):
    return self.eval_step(state)[0]


trials = 5
parameter_testing = {}
rollout_nums_mcts = 50
rollout_nums_mcts_ev = 250

env = rlcard.make("leduc-holdem")
env.game.init_game = types.MethodType(init_game, env.game)

cfr_agent = ("CFRAgent", models.load("leduc-holdem-cfr").agents[0])
random_agent = ("RandomAgent", RandomAgent(num_actions=env.num_actions))
mcts_ev_agent = ("MCTS with EV", MCTS_Expected(env, rollout_nums_mcts_ev, 0))
mcts_agent = ("MCTS", MCTS(env, rollout_nums_mcts, 0))
rule1_agent = ("Rule1 Agent", models.load("leduc-holdem-rule-v1").agents[0])
rule2_agent = ("Rule2 Agent", models.load("leduc-holdem-rule-v2").agents[0])
cfr_agent[1].step = types.MethodType(step, cfr_agent[1])

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()
mcts_both_rewards = []
env.set_agents(
    [MCTS_Expected(env, rollout_nums_mcts, 0), MCTS(env, rollout_nums_mcts, 1)]
)
for i in tqdm(range(250)):
    trajectories, payoffs = env.run(is_training=True)

    mcts_both_rewards.append(payoffs[0])
    # break
final_reward_both = [
    sum(mcts_both_rewards[: i + 1]) for i in range(len(mcts_both_rewards))
]

plt.plot(final_reward_both)
plt.xlabel("Episode")
plt.ylabel("MCTS EV Total Reward")
plt.title("MCTS EV vs MCTS")
plt.savefig("pics/MCTS_both.png")