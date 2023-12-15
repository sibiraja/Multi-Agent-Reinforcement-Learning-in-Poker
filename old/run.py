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