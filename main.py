""" A toy example of playing against pretrianed AI on Leduc Hold'em
"""
from tqdm import tqdm
from mcts import MCTS, TreeSearch, init_game

import rlcard
from rlcard import models
from rlcard.agents import LeducholdemHumanAgent as HumanAgent
from rlcard.agents import RandomAgent
from rlcard.utils import print_card
import types
from run import run

# Make environment
env = rlcard.make("leduc-holdem")
env.game.init_game = types.MethodType(init_game, env.game)

human_agent = HumanAgent(env.num_actions)
cfr_agent = models.load("leduc-holdem-cfr").agents[0]
random_agent = RandomAgent(num_actions=env.num_actions)
mcts_agent = MCTS(env, 100, 0)
# tree_agent = TreeSearch(env, {}, 1)
env.set_agents([mcts_agent, random_agent])

rewards_vs_random = []

print(">> Leduc Hold'em pre-trained model")

for i in tqdm(range(100)):
    trajectories, payoffs = env.run(is_training=True)

    # Let's take a look at what the agent card is

    rewards_vs_random.append(payoffs[0])
    # break
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()


plt.plot([sum(rewards_vs_random[: i + 1]) for i in range(len(rewards_vs_random))])
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.savefig("pics/vs_random.png")
