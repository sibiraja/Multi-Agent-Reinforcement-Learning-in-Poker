# same as MCTS except for the step method of MCTS_Expected

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

import types

from overrides import reset, init_game, run

# set global counter variables
trajectory = []
current_player = None

# Class used to conduct rollouts 
class TreeSearch:
    def __init__(self, env, state_nodes, player_id, first_player):
        self.env = env
        self.state_nodes = state_nodes
        self.player_id = player_id
        self.use_raw = False
        self.trajectory = []
        self.vector_to_string_S = {0: "SJ", 1: "SQ", 2: "SK"}
        self.vector_to_string = {0: "J", 1: "Q", 2: "K"}
        self.string_to_vector = {"J": (1, 0, 0), "Q": (0, 1, 0), "K": (0, 0, 1)}
        self.first_player = first_player

    def eval_step(self, state):
        # get perfect information state and format it correctly
        obs = self.env.get_perfect_information()

        # make sure that the hand card and number of chips has the current agent's values first
        if self.player_id == 1:
            obs["chips"] = [obs["chips"][1], obs["chips"][0]]
            obs["hand_cards"] = (obs["hand_cards"][1][1], obs["hand_cards"][0][1])
        else:
            obs["chips"] = [obs["chips"][0], obs["chips"][1]]
            obs["hand_cards"] = (obs["hand_cards"][0][1], obs["hand_cards"][1][1])

        if obs["public_card"] != None:
            obs["public_card"] = obs["public_card"][1]

        # Representation of our current state
        state = (
            tuple(obs["chips"]),
            obs["public_card"],
            tuple(obs["hand_cards"]),
            obs["current_round"],
        )
        
        # define some new variables for convenience
        own_card = self.string_to_vector[obs["hand_cards"][0]]
        opponent_card = self.string_to_vector[obs["hand_cards"][1]]
        chips = obs["chips"]
        public_card = obs["public_card"]
        new_chips = [0, 0]
        current_round = obs["current_round"]

        # calculate next rounds that each action will lead to, query for UCB score and append to action_UCB
        action_UCB = []
        new_round = None # define if an action will lead to a new round starting (public card revealed)
        legal_actions = obs["legal_actions"]
        for action in legal_actions:
            weights = self.probs(own_card, opponent_card)
            # generate a new public card (only used if necessary)
            new_public_card = np.random.choice(["J", "Q", "K"], p=weights) 

            # set chips to be same as opponents if call. Determine if public card must be generated
            if action == "call":
                if chips[0] != 1 and obs["current_round"] == 0:
                    new_round = True
                new_chips[0] = chips[1]
                if new_round == True:
                    next_state = (
                        (chips[1], new_chips[0]),
                        new_public_card,
                        (obs["hand_cards"][1], obs["hand_cards"][0]),
                        current_round + 1,
                    )
                    # if state has never been visited, add infinity. 
                    if self.state_nodes[next_state][2] != float("inf"):
                        action_UCB.append(-self.state_nodes[next_state][2])
                    else:
                        action_UCB.append(self.state_nodes[next_state][2])
                else:
                    next_state = (
                        (chips[1], new_chips[0]),
                        public_card,
                        (obs["hand_cards"][1], obs["hand_cards"][0]),
                        current_round,
                    )
                    # if state has never been visited, add infinity. 
                    if self.state_nodes[next_state][2] != float("inf"):
                        action_UCB.append(-self.state_nodes[next_state][2])
                    else:
                        action_UCB.append(self.state_nodes[next_state][2])

            # keep all chips the same, this will always start a new round if done in round 0.
            elif action == "check":
                if current_round == 0:
                    new_round = True

                if new_round == True:
                    next_state = (
                        (chips[1], chips[0]),
                        new_public_card,
                        (obs["hand_cards"][1], obs["hand_cards"][0]),
                        current_round + 1,
                    )
                    # if state has never been visited, add infinity. 
                    if self.state_nodes[next_state][2] != float("inf"):
                        action_UCB.append(-self.state_nodes[next_state][2])
                    else:
                        action_UCB.append(self.state_nodes[next_state][2])
                else:
                    next_state = (
                        (chips[1], chips[0]),
                        public_card,
                        (obs["hand_cards"][1], obs["hand_cards"][0]),
                        current_round,
                    )
                    # if state has never been visited, add infinity. 
                    if self.state_nodes[next_state][2] != float("inf"):
                        action_UCB.append(-self.state_nodes[next_state][2])
                    else:
                        action_UCB.append(self.state_nodes[next_state][2])

            # increase the number of chips you have in the pot
            elif action == "raise":
                if current_round == 0:
                    new_chips[0] = (
                        max([chips[0], chips[1]]) + self.env.game.round.raise_amount
                    )
                else:
                    new_chips[0] = (
                        max([chips[0], chips[1]]) + self.env.game.round.raise_amount
                    )

                next_state = (
                    (chips[1], new_chips[0]),
                    public_card,
                    (obs["hand_cards"][1], obs["hand_cards"][0]),
                    current_round,
                )
                # if state has never been visited, add infinity. 
                if self.state_nodes[next_state][2] != float("inf"):
                    action_UCB.append(-self.state_nodes[next_state][2])
                else:
                    action_UCB.append(self.state_nodes[next_state][2])

            # TreeSearch should never choose fold
            elif action == "fold":
                action_UCB.append(float("-inf"))

            else:
                raise Exception("Illegal action")

        # take action with highest UCB score
        take_action_index = np.argmax(action_UCB)
        take_action = legal_actions[take_action_index]

        reverse_action_mapping = {"call": 0, "raise": 1, "fold": 2, "check": 3}
        actual_action = reverse_action_mapping[take_action]

        # update trajectory with our current state
        global trajectory
        trajectory.append([state, self.player_id])
        global current_player
        current_player = self.player_id

        info = {take_action}
        return actual_action, info

    # same as eval_step
    def step(self, state):
        take_action_index, _ = self.eval_step(self, state)
        return take_action_index

    # allows querying the trajectory
    def return_trajectory(self):
        return self.trajectory

    # calculates the probabilities of the public card being any value
    def probs(self, state, opponent=[0, 0, 0]):
        probs = np.array([2, 2, 2])
        state = np.array(state)
        opponent = np.array(opponent)
        probs = probs - state[0:3] - opponent
        probs = probs / sum(probs)

        return list(probs)


#################################################################################

# Our actual agent class
class MCTS_Expected():

  def __init__(self, env, num_rollouts, player_id, model_path='./model'):
    self.env = env
    self.model_path = model_path
    self.num_rollouts = num_rollouts # number of TreeSearch rollouts to do before a given action
    self.player_id = player_id
    self.use_raw = False

    self.state_nodes = defaultdict(lambda: (0, 0, float("inf")))  # dictionary of rollout information
    self.vector_to_string_S = {0: 'SJ', 1: 'SQ', 2: 'SK'}
    self.vector_to_string = {0: 'J', 1: 'Q', 2: 'K'}
    self.string_to_vector = {'J': (1, 0, 0), 'Q': (0, 1, 0), 'K': (0, 0, 1)}

  # updates values at state_nodes after conducting a rollout
  def update_nodes(self, payoff):
    global trajectory
    for i in range(0, len(trajectory)):
      state = trajectory[i][0]
      current_player = trajectory[i][1]

      # update total_return and total_visits
      total_return, total_visits, UCB = self.state_nodes[state]

      if current_player == self.player_id:
        total_return += payoff
      else:
        total_return += -payoff

      total_visits += 1

      prev_state = trajectory[i-1][0] 
      _, prev_visits, _ = self.state_nodes[prev_state]

      # if not the root node, update the UCB score
      if i != 0:
        UCB = ((total_return + 7*total_visits)/14*total_visits) /total_visits - math.sqrt(np.log(prev_visits)/total_visits)

      self.state_nodes[state] = (total_return, total_visits, UCB)

  # given current state, take the next action (running through the game tree)
  def step(self, state):
    # extract the current (known) state of the game from the environment
    obs = self.env.get_state(self.player_id)
    obs = obs['obs']

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
    chips[1] = np.argmax(obs[21:36])

    # some hidden variables needed from the environment
    have_raised = self.env.game.round.have_raised
    not_raise_num = self.env.game.round.not_raise_num
    raise_amount = self.env.game.round.raise_amount
    allowed_raise_num = self.env.game.round.allowed_raise_num

    # variables we pass in to initialize the rollout environments
    state = {'current_player': self.player_id, 'public_card': public_card, 'hand': hand, 'all_chips': chips, 'have_raised': have_raised, 'not_raise_num': not_raise_num, 'raise_amount': raise_amount, 'allowed_raise_num': allowed_raise_num}

    for i in tqdm(range(self.num_rollouts)):
      #create env initialized to the given start state
      initialized_env = rlcard.make('leduc-holdem')
      initialized_env.run = types.MethodType(run, initialized_env)
      initialized_env.reset = types.MethodType(reset, initialized_env)
      initialized_env.game.init_game = types.MethodType(init_game, initialized_env.game)

      first_player = self.env.game.init_player == self.player_id

      my_player = TreeSearch(initialized_env, self.state_nodes, self.player_id, first_player)
      opponent = TreeSearch(initialized_env, self.state_nodes, 1-self.player_id, not first_player)

      # create the environment such that the agents have the correct player_id
      if self.player_id == 0:
        initialized_env.set_agents([my_player, opponent])
      else:
        initialized_env.set_agents([opponent, my_player])

      # Run a single rollout
      trajectories, payoffs = initialized_env.run(is_training=False, state = state)

      global trajectory
      global current_player 
      # add last game state to trajectory
      # only if last action wasn't fold --> 'action_record': [(0, 'raise'), (1, 'call')]
      if action_record[-1][1] != 'fold':
        temp = trajectories[self.player_id][-1]
        action_record = temp['action_record']

        final_state_obs = trajectories[1-current_player][-1]['raw_obs']

        my_card = final_state_obs['hand'][1]

        other_player_last_traj = trajectories[current_player][-1]['raw_obs']
        other_player_card = other_player_last_traj['hand'][1]
        my_tuple = (tuple(final_state_obs['all_chips']), final_state_obs['public_card'][1], (my_card, other_player_card), 1) # setting current round to 1 since it will always be 1 if the last action isn't fold
        ##prin("MY TUPLE: ", my_tuple)
        trajectory.append([my_tuple, 1-current_player])

      # Update state nodes
      self.update_nodes(payoffs[self.player_id])

      # clear trajectory
      trajectory = []

    # after finishing rollouts, decide what action to take call eval_step
    final_action_index, _ = self.eval_step(state)
    return final_action_index

  # take the next action, but do not do rollouts or update any nodes
  def eval_step(self, state):

    obs = self.env.get_state(self.player_id)
    temp = obs['legal_actions']
    obs = obs['obs']

    # extract legal actions at this state
    legal_actions = []
    for key in temp:
      legal_actions.append(key)

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

    # Initialize some values
    new_chips = [0, 0]
    card_values = ['J', 'Q', 'K']
    action_mapping = {0: "call", 1: "raise", 2: "fold", 3: "check"}
    new_round = False
    win_rates = np.array([0]*len(legal_actions))

    # Generate possible opponent cards and corresponding probabiltiies
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

    # iterate through possible opponent cards
    for j in range(0, len(possible_opponent_cards)):
        opponent_card = possible_opponent_cards[j]
        outer_win_rates = np.array([0]*len(legal_actions))

        # generate possible public cards and corresponding probabilities
        public_probs = self.probs(obs[0:3], opponent = self.string_to_vector[opponent_card])
        possible_public_cards = []
        public_card_weights = []
        for i in range(0, len(public_probs)):
          if public_probs[i] != 0:
            possible_public_cards.append(card_values[i])
            public_card_weights.append(public_probs[i])
        
        # iterate through possible public card
        for i in range(0, len(possible_public_cards)):
            new_public_card = possible_public_cards[i]
            inner_win_rates = []
            # calculate next state from any action, query state_nodes
            for action_number in legal_actions:

                action = action_mapping[action_number]

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

                elif action == "raise":
                    new_chips[0] = max([chips[0], chips[1]]) + self.env.game.round.raise_amount
                    next_state = ((chips[1], new_chips[0]), public_card, (opponent_card, own_card), current_round)
                    try:
                        inner_win_rates.append(-self.state_nodes[next_state][0]/self.state_nodes[next_state][1])
                    except ZeroDivisionError:
                        inner_win_rates.append(float("-inf"))

                elif action == "fold":
                    inner_win_rates.append(-chips[0]/2)

                else:
                    raise Exception("Illegal action")
            
            # scale values by probability weights
            scaled_inner_win_rates = public_card_weights[i] * np.array(inner_win_rates)
            outer_win_rates = outer_win_rates + scaled_inner_win_rates
        
        # scale values by probability weights
        scaled_outer_win_rates = opponent_card_weights[j] * outer_win_rates
        win_rates = win_rates + scaled_outer_win_rates

    # take action with highest "win_rate"
    final_action_index = np.argmax(win_rates)
    final_action = legal_actions[final_action_index]

    info = {}

    return final_action, info
  
  # calculates the probabilities of a drawn card being any value
  def probs(self, hand, opponent = [0, 0, 0], public = [0, 0, 0]):

    probs = np.array([2, 2, 2])
    probs = probs - hand - opponent - public
    probs = probs/sum(probs)

    return probs