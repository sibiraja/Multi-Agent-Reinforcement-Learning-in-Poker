''' A toy example of playing against pretrianed AI on Leduc Hold'em
'''

from mcts import MCTS, TreeSearch, init_game

import rlcard
from rlcard import models
from rlcard.agents import LeducholdemHumanAgent as HumanAgent
from rlcard.agents import RandomAgent
from rlcard.utils import print_card
import types
from run import run

# Make environment
env = rlcard.make('leduc-holdem')
env.game.init_game = types.MethodType(init_game, env.game)

human_agent = HumanAgent(env.num_actions)
cfr_agent = models.load('leduc-holdem-cfr').agents[0]
random_agent = RandomAgent(num_actions=env.num_actions)
mcts_agent = MCTS(env, 100, 0)
# tree_agent = TreeSearch(env, {}, 1)
env.set_agents([
    mcts_agent,
    human_agent
])



print(">> Leduc Hold'em pre-trained model")

while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=True)

    # print("Trajectories: ", trajectories)
    print()

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

    # Let's take a look at what the agent card is
    print('===============     Agent 1    ===============')
    print_card(env.get_perfect_information()['hand_cards'][1])
    print('===============     Agent 0    ===============')
    print_card(env.get_perfect_information()['hand_cards'][0])

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('Player 0 wins {} chips!'.format(payoffs[0]))
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('Player 0 lost {} chips!'.format(-payoffs[0]))
    print('')

    # break

    input("Press any key to continue...")