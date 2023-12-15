
from collections import defaultdict
import rlcard
import numpy as np
import math

from tqdm import tqdm

from rlcard import models
from rlcard.agents import LeducholdemHumanAgent as HumanAgent
from rlcard.utils import print_card
from rlcard.games.leducholdem import Dealer
from rlcard.games.leducholdem import Player
from rlcard.games.leducholdem import Judger
from rlcard.games.leducholdem import Round
from rlcard.games.limitholdem import Game
from rlcard.games.base import Card
import random
from run import run

'''

These functions initializes a new game either at a random starting state
or at a given state (which is useful when generating rollouts in MCTS)

'''

def step(self, state):
    return self.eval_step(state)[0]

def reset(self, state=None):

    if not state:

        state, player_id = self.game.init_game()
        self.action_recorder = []
        return self._extract_state(state), player_id

    else:

        state, player_id = self.game.init_game(state)
        self.action_recorder = []
        ##print("EXRACTED STATE INSIDE RESET: ", self._extract_state(state))
        return self._extract_state(state), player_id



def init_game(self, state=None):
    if not state:
        ''' Initialilze the game of Limit Texas Hold'em

        This version supports two-player limit texas hold'em

        Returns:
            (tuple): Tuple containing:

                (dict): The first state of the game
                (int): Current player's id
        '''
        # Initilize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)

        # Initilize two players to play the game
        self.players = [Player(i, self.np_random) for i in range(self.num_players)]

        # Initialize a judger class which will decide who wins in the end
        self.judger = Judger(self.np_random)

        # Prepare for the first round
        for i in range(self.num_players):
            self.players[i].hand = self.dealer.deal_card()
        # Randomly choose a small blind and a big blind
        s = self.np_random.randint(0, self.num_players)
        b = (s + 1) % self.num_players
        self.players[b].in_chips = self.big_blind
        self.players[s].in_chips = self.small_blind
        self.public_card = None
        # The player with small blind plays the first
        self.game_pointer = s

        # Initilize a bidding round, in the first round, the big blind and the small blind needs to
        # be passed to the round for processing.
        self.round = Round(raise_amount=self.raise_amount,
                           allowed_raise_num=self.allowed_raise_num,
                           num_players=self.num_players,
                           np_random=self.np_random)

        self.round.start_new_round(game_pointer=self.game_pointer, raised=[p.in_chips for p in self.players])

        # Count the round. There are 2 rounds in each game.
        self.round_counter = 0

        # Save the hisory for stepping back to the last state.
        self.history = []

        state = self.get_state(self.game_pointer)
        self.init_player = s
        return state, self.game_pointer

    else:
        ''' Initialilze the game of Limit Texas Hold'em

        This version supports two-player limit texas hold'em

        Returns:
            (tuple): Tuple containing:

                (dict): The first state of the game
                (int): Current player's id
        '''
        # Initilize a dealer that can deal cards
        ##print("INSIDE INIT GAME, STATE: ", state)

        self.dealer = Dealer(self.np_random)

        # Initilize two players to play the game
        self.players = [Player(i, self.np_random) for i in range(self.num_players)]

        # Initialize a judger class which will decide who wins in the end
        self.judger = Judger(self.np_random)

        s = state['current_player'] # s is mcts
        b = 1-s
        self.players[s].in_chips = state['all_chips'][0]
        self.players[b].in_chips = state['all_chips'][1]
        # self.players[b].in_chips = 2
        # self.players[s].in_chips = 4

        ##print("========CHIPS INSIDE INIT GAME: ", self.players[b].in_chips, self.players[s].in_chips)

        # The player with small blind plays the first
        self.init_player = s
        self.game_pointer = s
        # Prepare for the first round

        ##print()

        self.players[self.game_pointer].hand = Card(state['hand'][0], state['hand'][1])
        if not state['public_card']:
            self.public_card = None
        else:
            self.public_card = Card(state['public_card'][0], state['public_card'][1])
        self.players[b].hand = random.choice([card for card in [Card('S', 'J'), Card('H', 'J'), Card('S', 'Q'), Card('H', 'Q'), Card('S', 'K'), Card('H', 'K')] if (card not in [self.players[s].hand, self.public_card])])
        self.dealer.deck = [x for x in self.dealer.deck if x not in [self.players[s].hand, self.players[b].hand, self.public_card]]
        # Randomly choose a small blind and a big blind


        # Initilize a bidding round, in the first round, the big blind and the small blind needs to
        # be passed to the round for processing.
        self.round = Round(raise_amount=state['raise_amount'],
                           allowed_raise_num=state['allowed_raise_num'],
                           num_players=self.num_players,
                           np_random=self.np_random)

        self.round.start_new_round(game_pointer=self.game_pointer, raised=[p.in_chips for p in self.players])
        self.round.have_raised = state['have_raised']
        self.round.not_raise_num = state['not_raise_num']

        # Count the round. There are 2 rounds in each game.
        if not state['public_card']:
            self.round_counter = 0
        else:
            self.round_counter = 1

        # Save the hisory for stepping back to the last state.
        self.history = []

        state = self.get_state(self.game_pointer)

        ##print("========CHIPS INSIDE INIT GAME, ABOUT TO RETURN: ", self.players[b].in_chips, self.players[s].in_chips)
        ##prin("STATE INSIDE INIT GAME, ABOUT TO RETURN: ", state)

        return state, self.game_pointer


def run(self, is_training=False, state=None):
    '''
    Run a complete game, either for evaluation or training RL agent.

    Args:
        is_training (boolean): True if for training purpose.

    Returns:
        (tuple) Tuple containing:

            (list): A list of trajectories generated from the environment.
            (list): A list payoffs. Each entry corresponds to one player.

    Note: The trajectories are 3-dimension list. The first dimension is for different players.
            The second dimension is for different transitions. The third dimension is for the contents of each transiton
    '''
    trajectories = [[] for _ in range(self.num_players)]
    if state is None:
        state, player_id = self.reset()
    else:
        state, player_id = self.reset(state=state)

    # print(f"Legal actions inside run function: {self.get_perfect_information()['legal_actions']}")

    # Loop to play the game
    trajectories[player_id].append(state)
    while not self.is_over():
        # Agent plays
        if not is_training:
            action, _ = self.agents[player_id].eval_step(state)
        else:
            action = self.agents[player_id].step(state)


        # print(f"{player_id} is taking action with index: ", action)

        # Environment steps
        next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)
        # Save action
        trajectories[player_id].append(action)

        # Set the state and player
        state = next_state
        player_id = next_player_id

        # Save state.
        if not self.game.is_over():
            trajectories[player_id].append(state)

    # Add a final state to all the players
    for player_id in range(self.num_players):
        state = self.get_state(player_id)
        trajectories[player_id].append(state)

    # Payoffs
    payoffs = self.get_payoffs()

    return trajectories, payoffs


      # https://github.com/datamllab/rlcard/blob/7fc56edebe9a2e39c94f872edd8dbe325c61b806/rlcard/games/leducholdem/player.py
      # https://github.com/datamllab/rlcard/blob/master/rlcard/games/leducholdem/game.py#L138
      # https://github.com/datamllab/rlcard/blob/master/rlcard/envs/leducholdem.py
      # https://stackoverflow.com/questions/52292599/can-i-replace-an-existing-method-of-an-object-in-python