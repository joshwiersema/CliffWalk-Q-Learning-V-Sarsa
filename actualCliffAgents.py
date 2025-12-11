import numpy as np
import gymnasium as gym


def make_cliff_env(render_mode=None):
    """
    Helper to create the CliffWalking environment.
    """
    return gym.make("CliffWalking-v1", render_mode=render_mode)


def epsilon_greedy(Q, state, n_actions, epsilon):
    """
    ε-greedy policy over a tabular Q-table.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return int(np.argmax(Q[state]))


def train_q_learning(env,
                     num_episodes=500,
                     alpha=0.1,
                     gamma=0.99,
                     epsilon_start=1.0,
                     epsilon_end=0.1):
    """
    Tabular Q-Learning on a Gymnasium environment with discrete
    state and action spaces (like CliffWalking).
    
    Returns:
        Q: learned Q-table
        stats: dict with 'rewards', 'falls', 'steps'
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    rewards = []
    falls = []
    steps_list = []

    for ep in range(num_episodes):
        # Linear epsilon decay
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * ep / num_episodes
        )

        state, info = env.reset()
        done = False
        total_reward = 0.0
        fell = 0
        steps = 0

        while not done:
            action = epsilon_greedy(Q, state, n_actions, epsilon)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Off-policy target uses max over next state
            if not done:
                best_next = int(np.argmax(Q[next_state]))
                td_target = reward + gamma * Q[next_state, best_next]
            else:
                td_target = reward

            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            # Detect cliff fall (CliffWalking uses -100 for cliff)
            if reward <= -100:
                fell = 1

            state = next_state
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        falls.append(fell)
        steps_list.append(steps)

    stats = {
        "rewards": np.array(rewards, dtype=np.float32),
        "falls": np.array(falls, dtype=np.int32),
        "steps": np.array(steps_list, dtype=np.int32),
    }

    return Q, stats


def train_sarsa(env,
                num_episodes=500,
                alpha=0.1,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.1):
    """
    Tabular SARSA on a Gymnasium environment with discrete
    state and action spaces (like CliffWalking).
    
    Returns:
        Q: learned Q-table
        stats: dict with 'rewards', 'falls', 'steps'
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    rewards = []
    falls = []
    steps_list = []

    for ep in range(num_episodes):
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * ep / num_episodes
        )

        state, info = env.reset()
        action = epsilon_greedy(Q, state, n_actions, epsilon)

        done = False
        total_reward = 0.0
        fell = 0
        steps = 0

        while not done:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if not done:
                next_action = epsilon_greedy(Q, next_state, n_actions, epsilon)
                td_target = reward + gamma * Q[next_state, next_action]
            else:
                next_action = 0  # dummy
                td_target = reward

            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            if reward <= -100:
                fell = 1

            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        falls.append(fell)
        steps_list.append(steps)

    stats = {
        "rewards": np.array(rewards, dtype=np.float32),
        "falls": np.array(falls, dtype=np.int32),
        "steps": np.array(steps_list, dtype=np.int32),
    }

    return Q, stats


def extract_greedy_policy(Q, n_rows=4, n_cols=12):
    """
    Takes a Q-table and returns a (n_rows, n_cols) array of actions (0-3).
    Assumes CliffWalking layout (4x12).
    """
    n_states, n_actions = Q.shape
    assert n_states == n_rows * n_cols, "Q-table shape doesn't match grid size"
    greedy_actions = np.argmax(Q, axis=1).reshape((n_rows, n_cols))
    return greedy_actions


def print_policy_arrows(Q, n_rows=4, n_cols=12):
    """
    Pretty-print the greedy policy for CliffWalking as arrows.
    Hardcodes the typical S, G, and cliff cells for the 4x12 grid.
    """
    arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    greedy = extract_greedy_policy(Q, n_rows=n_rows, n_cols=n_cols)

    # Start and goal positions in the grid
    start_state = (n_rows - 1, 0)         # bottom-left
    goal_state = (n_rows - 1, n_cols - 1) # bottom-right

    print("Greedy policy (S = start, G = goal, C = cliff):")
    for r in range(n_rows):
        row_str = []
        for c in range(n_cols):
            # Cliff positions: bottom row, columns 1..(n_cols-2)
            if (r == n_rows - 1) and (1 <= c <= n_cols - 2):
                cell = "C"
            elif (r, c) == start_state:
                cell = "S"
            elif (r, c) == goal_state:
                cell = "G"
            else:
                cell = arrows[int(greedy[r, c])]
            row_str.append(cell)
        print(" ".join(row_str))

#print("Is this running?")
def compute_summary_stats(stats, last_n=50):
    """
    Compute simple summary statistics over the entire run and over
    the last N episodes.
    """
    rewards = stats["rewards"]
    falls = stats["falls"]
    steps = stats["steps"]

    summary = {}

    summary["mean_reward_all"] = float(np.mean(rewards))
    summary["mean_reward_last"] = float(np.mean(rewards[-last_n:]))

    summary["total_falls"] = int(np.sum(falls))
    summary["falls_rate_all"] = float(np.mean(falls))
    summary["falls_rate_last"] = float(np.mean(falls[-last_n:]))

    summary["mean_steps_all"] = float(np.mean(steps))
    summary["mean_steps_last"] = float(np.mean(steps[-last_n:]))

    return summary
