# pylint: disable=pointless-string-statement, redefined-outer-name, invalid-name, trailing-whitespace, wildcard-import

import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from vis_gym import *

BOLD = "\033[1m"  # ANSI escape sequence for bold text
RESET = "\033[0m"  # ANSI escape sequence to reset text formatting

train_flag = "train" in sys.argv
gui_flag = "gui" in sys.argv

setup(GUI=gui_flag)
env = game  # Gym environment already initialized within vis_gym.py

# env.render() # Uncomment to print game state info


def hash(obs):
    """
    Compute a unique compact integer ID representing the given observation.

    Encoding scheme:
      - Observation fields:
          * player_health: integer in {0, 1, 2}
          * window: a 3x3 grid of cells, indexed by (dx, dy) with dx, dy ∈ {-1, 0, 1}
          * guard_in_cell: optional identifier of a guard in the player's cell (e.g. 'G1', 'G2', ...)

      - Each cell contributes a single digit (0-8) to a base-9 number:
          * If the cell is out of bounds → code = 8
          * Otherwise:
                tile_type =
                    0 → empty
                    1 → trap
                    2 → heal
                    3 → goal
                has_guard = 1 if one or more guards present, else 0
                cell_value = has_guard * 4 + tile_type  # ranges from 0 to 7

        The 9 cell_values (row-major order: top-left → bottom-right) form a 9-digit base-9 integer `window_hash`.

      - The final state_id packs:
            * window_hash  → fine-grained local state
            * guard_index  → identity of guard in player's cell (0 if none, 1-4 otherwise)
            * player_health → coarse health component

        Specifically:
            WINDOW_SPACE = 9 ** 9
            GUARD_SPACE  = WINDOW_SPACE       # for guard_index (0-4)
            HEALTH_SPACE = GUARD_SPACE * 5    # for health (0-2)

            state_id = (player_health * HEALTH_SPACE)
                     + (guard_index * GUARD_SPACE)
                     + window_hash

    Returns:
        int: A unique, compact integer ID suitable for tabular RL (e.g. as a Q-table key).
    """
    health = int(obs.get("player_health", 0))
    window = obs.get("window", {})

    # Build cell values in a stable order: dx -1..1 (rows), dy -1..1 (cols)
    cell_values = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            cell = window.get((dx, dy))
            if cell is None or not cell.get("in_bounds", False):
                cell_values.append(8)
                continue

            # Determine tile type
            if cell.get("is_trap"):
                tile_type = 1
            elif cell.get("is_heal"):
                tile_type = 2
            elif cell.get("is_goal"):
                tile_type = 3
            else:
                tile_type = 0

            has_guard = 1 if cell.get("guards") else 0
            cell_value = has_guard * 4 + tile_type
            cell_values.append(cell_value)

    # Pack into base-9 integer
    window_hash = 0
    base = 1
    for v in cell_values:
        window_hash += v * base
        base *= 9

    # Include guard identity when player is in the center cell.
    # guard_in_cell is a convenience field set by the environment (e.g. 'G1' or None).
    guard_in_cell = obs.get("guard_in_cell")
    if guard_in_cell:
        # map 'G1' -> 1, 'G2' -> 2, etc.
        try:
            guard_index = int(str(guard_in_cell)[-1])
        except Exception:
            guard_index = 0
    else:
        guard_index = 0

    # window_hash uses 9^9 space; reserve an extra multiplier for guard identity (0..4)
    WINDOW_SPACE = 9**9
    GUARD_SPACE = WINDOW_SPACE  # one slot per guard id
    HEALTH_SPACE = (
        GUARD_SPACE * 5
    )  # 5 possible guard_id values (0 = none, 1-4 = guards)

    state_id = int(health) * HEALTH_SPACE + int(guard_index) * GUARD_SPACE + window_hash
    return state_id


"""
Complete the function below to do the following:

		1. Run a specified number of episodes of the game (argument num_episodes). An episode refers to starting in some initial
			 configuration and taking actions until a terminal state is reached.
		2. Maintain and update Q-values for each state-action pair encountered by the agent in a dictionary (Q-table).
		3. Use epsilon-greedy action selection when choosing actions (explore vs exploit).
		4. Update Q-values using the standard Q-learning update rule.

Important notes about the current environment and state representation

		- The environment is partially observable: observations returned by env.get_observation() include a centered 3x3
			"window" around the player plus the player's health. Each observation is a dict with these relevant keys:
					- 'player_position': (x, y)
					- 'player_health': integer (0=Critical, 1=Injured, 2=Full)
					- 'window': a dict keyed by (dx,dy) offsets in {-1,0,1} x {-1,0,1}. Each entry contains:
								{ 'guards': list or None, 'is_trap': bool, 'is_heal': bool, 'is_goal': bool, 'in_bounds': bool }
					- 'at_trap', 'at_heal', 'at_goal', and 'guard_in_cell' are convenience fields for the center cell.

		- To make a compact and consistent state hash for tabular Q-learning, encode the 3x3 window plus player health into a single integer.
			use the provided hash(obs) function above. Note that the player position is not included in the hash, as it is not needed for local decision-making.

		- Your Q-table should be a dict mapping state_id -> np.array of length env.action_space.n. Initialize arrays to zeros
			when you first encounter a state.

		- The actions available in this environment now include movement, combat, healing and waiting. The action indices are:
					0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: FIGHT, 5: HIDE, 6: HEAL, 7: WAIT

		- Remember to call obs, reward, done, info = env.reset() at the start of each episode.

		- Use a learning-rate schedule per (s,a) pair, i.e. eta = 1/(1 + N(s,a)) where N(s,a) is the
			number of updates applied to that pair so far.

Finally, return the dictionary containing the Q-values (called Q_table).
"""


def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
    """
        Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon should be decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
    Q_table = {}
    num_updates = {}
    episode_rewards = []

    for episode in tqdm(range(num_episodes)):
        # reset environment at the beginning of each episode
        obs, reward, done, info = env.reset()
        episode_reward = 0

        while not done:
            state = hash(obs)

            # Init Q-values for new state
            if state not in Q_table:
                Q_table[state] = np.zeros(env.action_space.n)
                num_updates[state] = np.zeros(env.action_space.n)

            # Action selection
            if np.random.random() < epsilon:
                # Choose random action
                action = env.action_space.sample()
            else:
                # Choose action with highest Q-value
                action = np.argmax(Q_table[state])

            # Take action, record next state and reward
            next_obs, reward, done, info = env.step(action)
            next_state = hash(next_obs)

            # Init Q-values for next state if new
            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(env.action_space.n)
                num_updates[next_state] = np.zeros(env.action_space.n)

            # Calculate Q-learning value
            eta = 1.0 / (1.0 + num_updates[state][action])
            V_old_next_state = np.max(Q_table[next_state])
            old_Q_value = Q_table[state][action]
            Q_table[state][action] = (1 - eta) * (old_Q_value) + eta * (
                reward + gamma * V_old_next_state
            )

            # Increment update number
            num_updates[state][action] += 1

            # Update observation & add episode reward
            obs = next_obs
            episode_reward += reward

            # Refresh GUI (if needed)
            if gui_flag:
                refresh(obs, reward, done, info, delay=0.1)

        # Store episode reward
        episode_rewards.append(episode_reward)

        # Decay epsilon
        epsilon *= decay_rate	

    # Plot rewards for each episode
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, alpha=0.2, color='blue', label='Episode Reward')
    
	# Calculate and plot running average for better visibility
    if num_episodes <= 1000:
        window_size = 50
    elif num_episodes <= 10000:
        window_size = 100
    elif num_episodes <= 100000:
        window_size = 500
    else:
        window_size = 1000
    
    if len(episode_rewards) >= window_size:
        cumulative_avg = np.cumsum(episode_rewards) / np.arange(1, len(episode_rewards) + 1)
        plt.plot(cumulative_avg, linewidth=2, color='green', label='Cumulative Average', linestyle='--')
    
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.title(f'Q-Learning Training Progress\n(Episodes={num_episodes}, Decay Rate={decay_rate}, γ={gamma})', fontsize=16)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'rewards_plot_{num_episodes}_{decay_rate}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nReward plot saved as: {plot_filename}")
    plt.close()
    
    return Q_table

"""
Specify number of episodes and decay rate for training and evaluation.
"""

num_episodes = 5000000
decay_rate = 0.9999995

"""
Run training if train_flag is set; otherwise, run evaluation using saved Q-table.
"""

if train_flag:
    Q_table = Q_learning(
        num_episodes=num_episodes, gamma=0.9, epsilon=1, decay_rate=decay_rate
    )  # Run Q-learning

    # Save the Q-table dict to a file
    with open(
        "Q_table_" + str(num_episodes) + "_" + str(decay_rate) + ".pickle", "wb"
    ) as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


"""
Evaluation mode: play episodes using the saved Q-table. Useful for debugging/visualization.
Based on autograder logic used to execute actions using uploaded Q-tables.
"""
def softmax(x, temp=1.0):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum(axis=0)

if not train_flag:
    rewards = []
    episode_lengths = []
    states_not_in_qtable = set()
    total_actions = 0
    actions_from_qtable = 0
    actions_random = 0

    action_counts = {
        'heal_state': np.zeros(env.action_space.n),
        'G1': np.zeros(env.action_space.n),
        'G2': np.zeros(env.action_space.n),
        'G3': np.zeros(env.action_space.n),
        'G4': np.zeros(env.action_space.n)
    }

    filename = "Q_table_" + str(num_episodes) + "_" + str(decay_rate) + ".pickle"
    input(
        f"\n{BOLD}Currently loading Q-table from "
        + filename
        + f"{RESET}.  \n\nPress Enter to confirm, or Ctrl+C to cancel and load a different Q-table file.\n(set num_episodes and decay_rate in Q_learning.py)."
    )
    Q_table = np.load(filename, allow_pickle=True)

    start_time = time.time()

    for episode in tqdm(range(10000)):
        obs, reward, done, info = env.reset()
        total_reward = 0
        steps = 0

        while not done:
            state = hash(obs)
            total_actions += 1

            # Check if state was seen during training
            if state in Q_table:
                actions_from_qtable += 1
                try:
                    # Select action using softmax over Q-values
                    action = np.random.choice(env.action_space.n, p=softmax(Q_table[state]))
                except KeyError:
                    # Fallback to random action if state not in Q-table
                    action = (env.action_space.sample())
            else:
                states_not_in_qtable.add(state)
                actions_random += 1
                action = env.action_space.sample()

            # Track actions taken in special situations
            if obs.get('at_heal', False):
                action_counts['heal_state'][action] += 1

            guard_in_cell = obs.get('guard_in_cell', None)
            if guard_in_cell and guard_in_cell in action_counts:
                action_counts[guard_in_cell][action] += 1

            obs, reward, done, info = env.step(action)

            total_reward += reward
            steps += 1
            if gui_flag:
                refresh(obs, reward, done, info, delay=0.1)  # Update the game screen [GUI only]

        rewards.append(total_reward)
        episode_lengths.append(steps)

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate metrics
    avg_reward = sum(rewards) / len(rewards)
    avg_episode_length = sum(episode_lengths) / len(episode_lengths)
    num_unique_states_in_qtable = len(Q_table)
    num_unseen_states = len(states_not_in_qtable)
    pct_actions_from_qtable = (actions_from_qtable / total_actions) * 100
    pct_actions_random = (actions_random / total_actions) * 100

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of unique states in Q-table: {num_unique_states_in_qtable}")
    print(f"Average reward over 10,000 episodes: {avg_reward:.2f}")
    print(f"Average episode length (actions): {avg_episode_length:.2f}")
    print(f"Total time for 10,000 episodes: {total_time:.2f} seconds")
    print(f"Unique states encountered NOT in Q-table: {num_unseen_states}")
    print(f"Percentage of actions from Q-table: {pct_actions_from_qtable:.2f}%")
    print(f"Percentage of random actions (unseen states): {pct_actions_random:.2f}%")
    print("="*60)

    situations = ['Heal State', 'Guard G1', 'Guard G2', 'Guard G3', 'Guard G4']
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FIGHT', 'HIDE', 'HEAL', 'WAIT']

    # Normalize each row to get distributions (handle division by zero)
    heatmap_data = []
    for situation in ['heal_state', 'G1', 'G2', 'G3', 'G4']:
        counts = action_counts[situation]
        total = counts.sum()
        if total > 0:
            normalized = counts / total
        else:
            normalized = np.zeros_like(counts)
        heatmap_data.append(normalized)

    heatmap_data = np.array(heatmap_data)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(action_names)))
    ax.set_yticks(np.arange(len(situations)))
    ax.set_xticklabels(action_names, fontsize=12)
    ax.set_yticklabels(situations, fontsize=12)

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Frequency', rotation=270, labelpad=20, fontsize=12)

    # Add text annotations
    for i in range(len(situations)):
        for j in range(len(action_names)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}', ha="center", va="center", color="black", fontsize=10)

    ax.set_xlabel('Actions', fontsize=14)
    ax.set_ylabel('Situation', fontsize=14)
    ax.set_title(f'Normalized Action Distribution by Situation\n(Episodes={num_episodes}, Decay Rate={decay_rate})', fontsize=16)
    plt.tight_layout()

    # Save heatmap
    heatmap_filename = f'actions_distribution_{num_episodes}_{decay_rate}.png'
    plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
    print(f"\nAction distribution heatmap saved as: {heatmap_filename}")
    plt.close()
