# Import Agents & Override Functions
from tqdm import tqdm
from mcts import MCTS, TreeSearch, init_game
from mcts_ev import MCTS_Expected
from mcts_nm import MCTS_NoMemory

import rlcard
from rlcard import models
from rlcard.agents import CFRAgent
from rlcard.agents import RandomAgent
from rlcard.agents import LeducholdemHumanAgent as HumanAgent
from rlcard.utils import print_card
import types
import numpy as np
from overrides import run, step

# Make environment
env = rlcard.make('leduc-holdem')
env.game.init_game = types.MethodType(init_game, env.game)

# Set hyperparameters
rollout_nums_mcts = 50
rollout_nums_mcts_ev = 250

# Initialize Agents
human_agent = HumanAgent(env.num_actions)
cfr_agent = models.load('leduc-holdem-cfr').agents[0]
random_agent = RandomAgent(num_actions=env.num_actions)
cfr_agent = ("CFRAgent", models.load("leduc-holdem-cfr").agents[0])
random_agent = ("RandomAgent", RandomAgent(num_actions=env.num_actions))
mcts_ev_agent = ("MCTS with EV", MCTS_Expected(env, rollout_nums_mcts_ev, 0))
mcts_agent = ("MCTS", MCTS(env, rollout_nums_mcts, 0))
rule1_agent = ("Rule1 Agent", models.load("leduc-holdem-rule-v1").agents[0])
rule2_agent = ("Rule2 Agent", models.load("leduc-holdem-rule-v2").agents[0])
cfr_agent[1].step = types.MethodType(step, cfr_agent[1])



# Note: change these lines to play with other agents. 
env.set_agents([mcts_agent[1], human_agent])



print(">> Leduc Hold'em pre-trained model")

while (True):
    print(">> Start a new game")

    # Run one game
    trajectories, payoffs = env.run(is_training=True)

    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    _action_list = []
    for i in range(1, len(action_record)+1):
        
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses', pair[1])

    # Let's take a look at what the agents card is
    print('===============     Agent 0    ===============')
    print_card(env.get_perfect_information()['hand_cards'][0])
    print('===============     Agent 1   ===============')
    print_card(env.get_perfect_information()['hand_cards'][1])

    print('===============     Result     ===============')

    # Print Player 0's Payoff
    if payoffs[0] > 0:
        print('Player 0 wins {} chips!'.format(payoffs[0]))
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('Player 0 lost {} chips!'.format(-payoffs[0]))
    print('')

    # break

    input("Press any key to continue...")