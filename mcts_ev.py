
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




      # https://github.com/datamllab/rlcard/blob/7fc56edebe9a2e39c94f872edd8dbe325c61b806/rlcard/games/leducholdem/player.py
      # https://github.com/datamllab/rlcard/blob/master/rlcard/games/leducholdem/game.py#L138
      # https://github.com/datamllab/rlcard/blob/master/rlcard/envs/leducholdem.py
      # https://stackoverflow.com/questions/52292599/can-i-replace-an-existing-method-of-an-object-in-python

import types

trajectory = []
current_player = None

class TreeSearch():

  def __init__(self, env, state_nodes, player_id, first_player):
    self.env = env
    self.state_nodes = state_nodes
    self.player_id = player_id
    self.use_raw = False
    self.trajectory = []
    self.vector_to_string_S = {0: 'SJ', 1: 'SQ', 2: 'SK'}
    self.vector_to_string = {0: 'J', 1: 'Q', 2: 'K'}
    self.string_to_vector = {'J': (1, 0, 0), 'Q': (0, 1, 0), 'K': (0, 0, 1)}
    self.first_player = first_player

  def eval_step(self, state):
    # get overall state

    obs = self.env.get_perfect_information()
    ##prin("GETTING STATE", self.env.get_state(self.player_id))
    ##prin("INSIDE TREESEARCH, BEFORE OBS WAS ", obs)

    # make sure that the hand card and number of chips has our agent's values first
    if self.player_id == 1:
      obs['chips'] = [obs['chips'][1], obs['chips'][0]]
      obs['hand_cards'] = (obs['hand_cards'][1][1], obs['hand_cards'][0][1])
    else:
      ##prin("PRINT: ", obs['chips'])
      # obs['chips'] = (obs['chips'][0][1], obs['chips'][1][1])
      obs['chips'] = [obs['chips'][0], obs['chips'][1]]
      obs['hand_cards'] = (obs['hand_cards'][0][1], obs['hand_cards'][1][1])

    if obs['public_card'] != None:
      obs['public_card'] = obs['public_card'][1]

    state = (tuple(obs['chips']), obs['public_card'], tuple(obs['hand_cards']), obs['current_round'])
    ##prin("INSIDE TREESEARCH, AFTER OBS WAS ", obs)
    ##prin("STATE FROM TREESEARCH: ", state)

    legal_actions = obs['legal_actions']
    ##prin("Legal actions inside Treesearch are: ", legal_actions)

    # take the action with the highest UCB score

    own_card = self.string_to_vector[obs['hand_cards'][0]]
    opponent_card = self.string_to_vector[obs['hand_cards'][1]]
    """
    action_values = []
    for action in legal_actions:
      action_values.append(self.state_nodes[state])

    take_action_index = np.argmax(action_values[2])
    take_action = legal_actions[take_action_index]
    """
    chips = obs['chips']
    public_card = obs['public_card']

    action_UCB = []
    new_round = None

    new_chips = [0, 0]
    current_round = obs['current_round']

    # ##prin("UCB FOR CURRENT STATE: ", self.state_nodes)
    # for key, value in self.state_nodes.items():
    #   ##prin("KEY: ", key)
    #   ##prin("VALUE: ", value)


    for action in legal_actions:
      weights = self.probs(own_card, opponent_card)
      new_public_card = np.random.choice(['J', 'Q', 'K'], p = weights)

      if action == "call":
        if chips[0] != 1 and obs['current_round'] == 0:
          new_round = True


        new_chips[0] = chips[1]
        if new_round == True:

          next_state = ((chips[1], new_chips[0]), new_public_card, (obs['hand_cards'][1], obs['hand_cards'][0]), current_round + 1)
          ##prin("Next state for choosing call inside Treesearch: ", next_state)
          if self.state_nodes[next_state][2] != float("inf"):
            action_UCB.append(-self.state_nodes[next_state][2])
          else:
            action_UCB.append(self.state_nodes[next_state][2])
        else:
          next_state = ((chips[1],new_chips[0]), public_card, (obs['hand_cards'][1], obs['hand_cards'][0]), current_round)
          ##prin("Next state for choosing call inside Treesearch: ", next_state)
          # action_UCB.append(-self.state_nodes[next_state][2])
          if self.state_nodes[next_state][2] != float("inf"):
              action_UCB.append(-self.state_nodes[next_state][2])
          else:
            action_UCB.append(self.state_nodes[next_state][2])
       #prin("Treesearch querying CALL:", next_state)


      elif action == "check":
        if current_round == 0:
          new_round = True

        if new_round == True:
          next_state = ((chips[1], chips[0]), new_public_card, (obs['hand_cards'][1], obs['hand_cards'][0]), current_round + 1)
          # action_UCB.append(-self.state_nodes[next_state][2])
          if self.state_nodes[next_state][2] != float("inf"):
              action_UCB.append(-self.state_nodes[next_state][2])
          else:
            action_UCB.append(self.state_nodes[next_state][2])
        else:
          next_state = ((chips[1], chips[0]), public_card, (obs['hand_cards'][1], obs['hand_cards'][0]), current_round)
          # action_UCB.append(-self.state_nodes[next_state][2])
          if self.state_nodes[next_state][2] != float("inf"):
              action_UCB.append(-self.state_nodes[next_state][2])
          else:
            action_UCB.append(self.state_nodes[next_state][2])
       #prin("Treesearch querying CHECK:", next_state)

      elif action == "raise":
        if current_round == 0:
          new_chips[0] = max([chips[0], chips[1]]) + self.env.game.round.raise_amount
        else:
          new_chips[0] = max([chips[0], chips[1]]) + self.env.game.round.raise_amount

        next_state = ((chips[1], new_chips[0]), public_card, (obs['hand_cards'][1], obs['hand_cards'][0]), current_round)
        ##prin("Next state for choosing raise inside Treesearch: ", next_state)
        # action_UCB.append(-self.state_nodes[next_state][2])
        if self.state_nodes[next_state][2] != float("inf"):
          action_UCB.append(-self.state_nodes[next_state][2])
        else:
          action_UCB.append(self.state_nodes[next_state][2])
       #prin("Treesearch querying RAISE:", next_state)

      elif action == "fold":
        # action_UCB.append(-chips[0]/2)
        action_UCB.append(float("-inf"))

      else:
        raise Exception("Illegal action")
    ##prin("ACTION UCB: ", action_UCB)

    take_action_index = np.argmax(action_UCB)
    take_action = legal_actions[take_action_index]

    reverse_action_mapping = {"call": 0, "raise": 1, "fold": 2, "check": 3}
    actual_action = reverse_action_mapping[take_action]

    """
    action_values = []
    for action in legal_actions:
      action_values.append(self.state_nodes[state])

    take_action_index = np.argmax(action_values[2])
    take_action = legal_actions[take_action_index]
    """
    # #PLACEHOLDER: Take a random action
    # take_action_index = np.random.choice(range(0,len(legal_actions)))
    # take_action = legal_actions[take_action_index]

    # before running this, we need a global trajectory
    global trajectory
    trajectory.append([state, self.player_id])
    ##prin("TRAJECTORY FROM TREESEARCH: ", trajectory)

    info = {take_action}

    ##prin(f"Treesearch {self.player_id} is taking action: ", take_action)
    global current_player
    current_player = self.player_id

    return actual_action, info
    # return take_action_index

  # same as eval_step
  def step(self, state):
    take_action_index, info = self.eval_step(self, state)
    return takle_action_index

  def return_trajectory(self):
    return self.trajectory

  def probs(self, state, opponent = [0, 0, 0]):


    probs = np.array([2, 2, 2])
    state = np.array(state)
    ##prin("STATE: ", state)
    opponent = np.array(opponent)
    ##prin("OPPONENT: ", opponent)
    ##prin("Probs: ", probs)
    probs = probs - state[0:3] - opponent
    probs = probs/sum(probs)

    return list(probs)



#################################################################################
class MCTS_Expected():

  def __init__(self, env, num_rollouts, player_id, model_path='./model'):
    self.env = env
    self.model_path = model_path
    self.num_rollouts = num_rollouts
    self.player_id = player_id
    self.use_raw = False

    self.state_nodes = defaultdict(lambda: (0, 0, float("inf")))
    self.vector_to_string_S = {0: 'SJ', 1: 'SQ', 2: 'SK'}
    self.vector_to_string = {0: 'J', 1: 'Q', 2: 'K'}
    self.string_to_vector = {'J': (1, 0, 0), 'Q': (0, 1, 0), 'K': (0, 0, 1)}

  # updates values at state_nodes from a trajectory
  def update_nodes(self, payoff):
    global trajectory
    for i in range(0, len(trajectory)):
      state = trajectory[i][0]
      current_player = trajectory[i][1]

      total_return, total_visits, UCB = self.state_nodes[state]

      if current_player == self.player_id:
        total_return += payoff
      else:
        total_return += -payoff

      total_visits += 1

      prev_state = trajectory[i-1][0] # ((2, 2), 'K', ('J', 'J'), 1), 0
      ##prin("PREV STATE: ", prev_state)
      ##prin("STATE NODES: ", self.state_nodes)
      _, prev_visits, _ = self.state_nodes[prev_state]
      ##prin("PREV VISITS: ", prev_visits)

      # if not the root node, update the UCB score
      if i != 0:
        first_term = total_return/total_visits
        second_term = math.sqrt(np.log(prev_visits)/total_visits)
        UCB = ((total_return + 7*total_visits)/14*total_visits) /total_visits - math.sqrt(np.log(prev_visits)/total_visits)

      self.state_nodes[state] = (total_return, total_visits, UCB)

  # given current state, take the next action (running through the game tree)
  def step(self, state):
    # get current state
    obs = self.env.get_state(self.player_id)
    obs = obs['obs']

    current_player_id = self.player_id
    if max(obs[3:6]) == 0:
      public_card = None
    else:
      public_card = self.vector_to_string_S[np.argmax(obs[3:6])]

    if max(obs[0:3]) == 0:
      raise Exception("We don't have a card?")
    else:
      hand = self.vector_to_string_S[np.argmax(obs[0:3])]

    chips = [0, 0]
    chips[0] = np.argmax(obs[6:21])
    ##prin("OBS[6:21]: ", obs[6:21])
    ##prin("CHIPS[0]: ", chips[0])
    chips[1] = np.argmax(obs[21:36])
    ##prin("OBS[21:36]: ", obs[21:36])
    ##prin("CHIPS[1]: ", chips[1])


    have_raised = self.env.game.round.have_raised
    not_raise_num = self.env.game.round.not_raise_num

    ##prin("HAVE RAISED: ", have_raised)
    ##prin("NOT RAISE NUM: ", not_raise_num)

    raise_amount = self.env.game.round.raise_amount
    allowed_raise_num = self.env.game.round.allowed_raise_num

    state = {'current_player': self.player_id, 'public_card': public_card, 'hand': hand, 'all_chips': chips, 'have_raised': have_raised, 'not_raise_num': not_raise_num, 'raise_amount': raise_amount, 'allowed_raise_num': allowed_raise_num}
    ##prin("INITIAL STATE: ", state)

    for i in tqdm(range(self.num_rollouts)):
      #create env initialized to the given start state (randomize the card for the opponent player)
      initialized_env = rlcard.make('leduc-holdem')
      initialized_env.run = types.MethodType(run, initialized_env)
      initialized_env.reset = types.MethodType(reset, initialized_env)
      initialized_env.game.init_game = types.MethodType(init_game, initialized_env.game)


      # obs, player_id = initialized_env.reset(state = state)
      ##prin("OBS AFTER INITIALIZING: ", obs)
      ##prin("PLAYER ID AFTER INITIALIZING: ", player_id)

      first_player = self.env.game.init_player == self.player_id

      my_player = TreeSearch(initialized_env, self.state_nodes, self.player_id, first_player)
      opponent = TreeSearch(initialized_env, self.state_nodes, 1-self.player_id, not first_player)

      # create the environment such that we have the correct player_id
      if self.player_id == 0:
        initialized_env.set_agents([my_player, opponent])
      else:
        initialized_env.set_agents([opponent, my_player])

      # Run a single rollout
      trajectories, payoffs = initialized_env.run(is_training=False, state = state)
      ##prin('LAST TRAJECTORY', trajectories[self.player_id][-1])

      temp = trajectories[self.player_id][-1]

      ##prin(type(temp)) # this is a dict

      # for keys, values in temp.items():
      #  #prin("keys: ", keys)
      #  #prin("values: ", values)

      ##prin("=========")

      action_record = temp['action_record']

      # Update values at nodes
      global trajectory
      global current_player 
      ##prin("TRAJECTORY FROM MCTS: ", trajectory)

      final_state_obs = trajectories[1-current_player][-1]['raw_obs']

      ##prin("LAST STATE OBS: ", final_state_obs)
      my_card = final_state_obs['hand'][1]

      other_player_last_traj = trajectories[current_player][-1]['raw_obs']
      other_player_card = other_player_last_traj['hand'][1]

    # only append if last action wasn't fold --> 'action_record': [(0, 'raise'), (1, 'call')]
      if action_record[-1][1] != 'fold':
        ##prin("APPENDING TO TRAJECTORY")
        # trajectory.append([((chips[1], chips[0]), public_card, (other_player_card, my_card), 1), 1-self.player_id])
        my_tuple = (tuple(final_state_obs['all_chips']), final_state_obs['public_card'][1], (my_card, other_player_card), 1) # setting current round to 1 since it will always be 1 if the last action isn't fold
        ##prin("MY TUPLE: ", my_tuple)
        trajectory.append([my_tuple, 1-current_player])

      self.update_nodes(payoffs[self.player_id])

      ##prin("=========")
     #prin("=============ROLLOUT ENDED==============")
      trajectory = []


    ##prin("STATE NODES: ", self.state_nodes)

    # after finishing rollouts, decide what action to take (same as eval_step)
    final_action_index, info = self.eval_step(state)
    return final_action_index

  # take the next action, but do not do rollouts or update any nodes
  def eval_step(self, state):

    obs = self.env.get_state(self.player_id)
    temp = obs['legal_actions']
    ##prin(type(legal_actions))
    ##prin("Legal actions inside MCTS are: ", legal_actions)
    obs = obs['obs']

    # temp2 = []
    legal_actions = []

    for key in temp:
      legal_actions.append(key)

    # for key in temp:
    #   for i in range(len(key)):
    #     temp2.append(key[0][i])

    # for i in range(0, len(temp2)):
    #   legal_actions.append(temp2[i][0])

    ##prin(type(legal_actions))
    ##prin("Legal actions inside MCTS are: ", legal_actions)

    # figure out your card
    if max(obs[0:3]) == 0:
      raise Exception("We don't have a card?")
    else:
      own_card = self.vector_to_string[np.argmax(obs[0:3])]

    # figure out the public card
    if max(obs[3:6]) == 0:
      public_card = None
    else:
      public_card = self.vector_to_string[np.argmax(obs[3:6])]

    # figure out the current round
    if public_card == None:
      current_round = 0
    else:
      current_round = 1

    # figure out the number of chips in the pot
    chips = [0, 0]
    chips[0] = np.argmax(obs[6:21])
    chips[1] = np.argmax(obs[21:36])


    first_player = (self.env.game.init_player == self.player_id)
    ##prin("LEGAL ACTIONS: ", legal_actions)
    ##prin(type(legal_actions))

   #prin("CURRENT STATE:", state)

    new_chips = [0, 0]

    # maximize over average return of taking any action at root node
    card_values = ['J', 'Q', 'K']
    action_mapping = {0: "call", 1: "raise", 2: "fold", 3: "check"}
    new_round = False
    win_rates = []

    # Generate possible opponent card
    if public_card == None:
        opponent_probs = self.probs(obs[0:3])
    else:
       opponent_probs = self.probs(obs[0:3], public = obs[3:6])

    possible_opponent_cards = []
    opponent_card_weights = []
    for i in range(0, len(opponent_probs)):
        if opponent_probs[i] != 0:
            possible_opponent_cards.append(card_values[i])
            opponent_card_weights.append(opponent_probs[i])
   #prin("Possible opponent cards:", possible_opponent_cards)

    win_rates = np.array([0]*len(legal_actions))
    for j in range(0, len(possible_opponent_cards)):
        opponent_card = possible_opponent_cards[j]
        outer_win_rates = np.array([0]*len(legal_actions))
        # generate possible public cards
        public_probs = self.probs(obs[0:3], opponent = self.string_to_vector[opponent_card])
        possible_public_cards = []
        public_card_weights = []
        for i in range(0, len(public_probs)):
          if public_probs[i] != 0:
            possible_public_cards.append(card_values[i])
            public_card_weights.append(public_probs[i])
        
        for i in range(0, len(possible_public_cards)):
            new_public_card = possible_public_cards[i]
            inner_win_rates = []
            for action_number in legal_actions:
            # action_number = action[0]

                action = action_mapping[action_number]
                    ##prin("ACTION: ", action)

                if action == "call":
                    if chips[0] != 1 and current_round == 0:
                        new_round = True

                    new_chips[0] = chips[1]
                    if new_round == True:

                        next_state = ((chips[1], new_chips[0]), new_public_card, (opponent_card, own_card), current_round+1)
                        try:
                            inner_win_rates.append(-self.state_nodes[next_state][0]/self.state_nodes[next_state][1])
                        except ZeroDivisionError:
                            inner_win_rates.append(float("-inf"))
                    else:
                        next_state = ((chips[1], new_chips[0]), public_card, (opponent_card, own_card), current_round)
                        try:
                            inner_win_rates.append(-self.state_nodes[next_state][0]/self.state_nodes[next_state][1])
                        except ZeroDivisionError:
                            inner_win_rates.append(float("-inf"))
                   #print("QUERYING CALL, NEXT STATE:", next_state)

                ##print("NEXT STATE: ", next_state)

            # elif action == "check":
            #   new_round = True
            #   weights = self.probs(obs, self.string_to_vector[opponent_card])
            #   public_card = np.random.choice(['J', 'Q', 'K'], p = weights)
            # #   next_state = ((chips[1], chips[0]), public_card, (opponent_card, own_card), current_round + 1)
            #   try:
            #     win_rates.append(-self.state_nodes[next_state][0]/self.state_nodes[next_state][1])
            #   except ZeroDivisionError:
            #     win_rates.append(float("-inf"))

                elif action == "check":
                    if current_round == 0:
                        new_round = True

                    if new_round == True:
                        next_state = ((chips[1], chips[0]), new_public_card, (opponent_card, own_card), current_round + 1)
                    else:
                        next_state = ((chips[1], chips[0]), public_card, (opponent_card, own_card), current_round)
                    try:
                        inner_win_rates.append(-self.state_nodes[next_state][0]/self.state_nodes[next_state][1])
                    except ZeroDivisionError:
                        inner_win_rates.append(float("-inf"))
                   #print("QUERYING CHECK, NEXT STATE:", next_state)

                elif action == "raise":
                    new_chips[0] = max([chips[0], chips[1]]) + self.env.game.round.raise_amount
                    next_state = ((chips[1], new_chips[0]), public_card, (opponent_card, own_card), current_round)
                    try:
                        inner_win_rates.append(-self.state_nodes[next_state][0]/self.state_nodes[next_state][1])
                    except ZeroDivisionError:
                        inner_win_rates.append(float("-inf"))
                   #print("QUERYING RAISE, NEXT STATE:", next_state)

                elif action == "fold":
                    ##print("==CHOOSING TO FOLD==")
                    inner_win_rates.append(-chips[0]/2)

                else:
                    raise Exception("Illegal action")
                
            scaled_inner_win_rates = public_card_weights[i] * np.array(inner_win_rates)
            outer_win_rates = outer_win_rates + scaled_inner_win_rates

        scaled_outer_win_rates = opponent_card_weights[j] * outer_win_rates
        win_rates = win_rates + scaled_outer_win_rates
        # if action != "fold":
            ##print("NEXT STATE :" , next_state)

   #print("WIN RATES: ", win_rates)

    final_action_index = np.argmax(win_rates)
   #print("FINAL ACTION INDEX: ", final_action_index)
    final_action = legal_actions[final_action_index]
   #print("FINAL ACTION: ", final_action)

    # reverse_action_mapping = {"call": 0, "raise": 1, "fold": 2, "check": 3}

    info = {}

    return final_action, info
    # return final_action_index

  def probs(self, hand, opponent = [0, 0, 0], public = [0, 0, 0]):

    probs = np.array([2, 2, 2])
    probs = probs - hand - opponent - public
    probs = probs/sum(probs)

    return probs
