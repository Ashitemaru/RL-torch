import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, ".."))

from utils.policy_wrapper import Numpy1DArrayPolicy
from utils.run_gym import run_environment


def epsilon_greedy_policy(n_state, n_action, q_function, eps=0.5):
    policy = np.ones((n_state, n_action)) * (eps / n_action)
    for state in range(n_state):
        action = np.argmax(q_function[state])
        policy[state][action] += 1 - eps

    return policy


def sample_action(policy, state):
    return np.random.choice(a=np.array(range(policy.shape[1])), p=policy[state])


def q_learning_step(
    q_function, state, action, reward, next_state, next_action, terminal, alpha, gamma
):
    next_q_function = q_function
    next_q_function[state][action] += alpha * (
        reward + gamma * max(q_function[next_state]) - q_function[state][action]
    )

    return next_q_function


def sarsa_step(
    q_function, state, action, reward, next_state, next_action, terminal, alpha, gamma
):
    next_q_function = q_function
    next_q_function[state][action] += alpha * (
        reward + gamma * q_function[next_state][next_action] - q_function[state][action]
    )

    return next_q_function


def learn(learning_step, episodes=5000, max_steps=100, alpha=0.8, gamma=0.9):
    """
    The integrated learning function for Q learning & Sarsa.
    """

    # This env is the one agents will interact with, which is called **training env**
    env = gym.make("FrozenLake-v1", render_mode="ansi", is_slippery=False)
    env = TimeLimit(env, max_steps)

    n_state = env.observation_space.n
    n_action = env.action_space.n
    q_function = np.zeros((n_state, n_action))

    # To satisfy the constriction of GLIE
    eps_annaling = 1 / episodes

    for episode in range(episodes):
        state, _ = env.reset()
        terminal, truncated = False, False

        while True:
            # Policy improvement of GPI
            # The behavior policies of Q learning & Sarsa are all epsilon greedy policy
            policy = epsilon_greedy_policy(
                n_state, n_action, q_function, eps=1 - episode * eps_annaling
            )
            action = sample_action(policy, state)

            # Run the behavior policy to step the env
            next_state, reward, terminal, truncated, _ = env.step(action)

            # Policy evaluation of GPI
            # The target policy of Q learning is fully greedy policy (off-policy)
            # The target policy of Sarsa is the same as behavior policy (on-policy)
            next_action = sample_action(policy, next_state)
            q_function = learning_step(
                q_function,
                state,
                action,
                reward,
                next_state,
                next_action,
                terminal,
                alpha,
                gamma,
            )

            state = next_state

            if terminal or truncated:
                break

    # After iteration converages, return the optim policy according to the optim Q function
    return q_function, q_function.argmax(axis=1)


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)

    print("\n" + "-" * 25 + "\nBeginning Q Learning\n" + "-" * 25)
    _, policy_q_learning = learn(q_learning_step, alpha=0.8, gamma=0.9)
    run_environment(
        env=env,
        policy=Numpy1DArrayPolicy(policy_q_learning),
        path="../../image/q_learning.gif",
        fps=10,
    )

    print("\n" + "-" * 25 + "\nBeginning Sarsa\n" + "-" * 25)
    _, policy_sarsa = learn(sarsa_step, alpha=0.8, gamma=0.9)
    run_environment(
        env=env,
        policy=Numpy1DArrayPolicy(policy_sarsa),
        path="../../image/sarsa.gif",
        fps=10,
    )
