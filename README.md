# Determinization-Based Monte Carlo Tree Search in Leduc Poker

Multi-Agent Reinforecement Learning: this work explores  the strength of different variations of Perfect Information Monte Carlo Tree Search when playing Leduc Poker, a simplified version of traditional Texas Holdem poker. It uses the RLCard environment provided by https://github.com/datamllab/rlcard.

## Usage

### Installation
This codebase uses `python3`. Create a virtual environment by running `python3 -m venv .venv` and then activate it with `source venv/bin/activate`. Then, run `pip install -r requirements.txt` to install the required packages.


### Codebase Structure
The codebase is organized as follows:
- `main.py`: The main entrypoint for the codebase to play agents against one another or play against agents as a human agent
- `mcts.py`: The implementation of the vanilla MCTS algorithm, choosing actions by generating an opponent and public card at random
- `mcts_ev.py`: The implementation of the MCTS EV algorithm, which chooses actions according to the expected value of reward produced by each action based on all possible opponent and public card combinations
- `mcts_nm.py`: The implementation of the MCTS NM algorithm, which chooses actions similar to MCTS EV, but resets learnt UCB scores between rounds
- `overrides.py`: The implementation of the overrides for the RLCard environment to allow for the use of custom MCTS agents and the assumptions that MCTS requires
- `hyperparameters.py`: Plotting script to do hyperparameter sweeps
- `plot_rewards.py`: Plotting script to plot the rewards of agents against one another

### Running Experiments
To play agents against one another live, or to play against an agent manually as a human agent, simply change the line `env.set_agents([mcts_agent[1], human_agent])` in `main.py` to the desired agents. Then, run `python main.py` to play the game.

Currently, these agents have been tested:
- `human_agent`
- `cfr_agent `
- `random_agent`
- `cfr_agent`
- `random_agent`
- `mcts_ev_agent`
- `mcts_agent`
- `rule1_agent`
- `rule2_agent`

Note that more agents can be tested by referring to the RLCard documentation and adding them to `main.py` in the same manner. 

To run a hyperparameter sweep, run `python hyperparameters.py`. This will run a hyperparameter sweep over the number of rollouts for the desired MCTS agents, which can be specified in the same manner as above. To plot the performance of agents against one another, run `python plot_rewards.py` after having specified the desired agents in the same manner as the previous two scripts.