"""
Buggy code warning!

The implementation of Off-policy Monte-Carlo Control may be buggy.
The detailed note can be found at a @todo comment in the code.
May be fixed later, but not that fast.
"""

import os
import sys
import numpy as np
from tqdm import tqdm
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


def mc_on_policy_step(trajectory, q_function, alpha, gamma):
    next_q_function = q_function
    accumulated_reward = 0
    for state, action, reward in reversed(trajectory):
        accumulated_reward = accumulated_reward * gamma + reward
        next_q_function[state][action] += alpha * (
            accumulated_reward - q_function[state][action]
        )

    return next_q_function


def mc_off_policy_step(
    trajectory, q_function, target_policy, behavior_policy, alpha, gamma
):
    next_q_function = q_function
    accumulated_reward = 0
    sample_weight = 1
    for state, action, reward in reversed(trajectory):
        accumulated_reward = accumulated_reward * gamma + reward
        sample_weight *= target_policy[state][action] / behavior_policy[state][action]
        next_q_function[state][action] += alpha * (
            accumulated_reward * sample_weight - q_function[state][action]
        )

    return next_q_function


def learn(on_policy, episodes=5000, max_steps=100, alpha=0.8, gamma=0.9):
    # This env is the one agents will interact with, which is called **training env**
    env = gym.make("FrozenLake-v1", render_mode="ansi", is_slippery=False)
    env = TimeLimit(env, max_steps)

    n_state = env.observation_space.n
    n_action = env.action_space.n
    q_function = np.zeros((n_state, n_action))

    # To satisfy the constriction of GLIE
    eps_annaling = 1 / episodes

    for episode in tqdm(range(episodes)):
        state, _ = env.reset()
        terminal, truncated = False, False

        # Policy improvement of the MC Control
        trajectory = []
        policy = epsilon_greedy_policy(
            n_state, n_action, q_function, eps=1 - episode * eps_annaling
        )

        # Sample a full trajectory
        while True:
            action = sample_action(policy, state)

            # Run the behavior policy to step the env
            next_state, reward, terminal, truncated, _ = env.step(action)
            trajectory.append((state, action, reward))

            state = next_state

            if terminal or truncated:
                break

        # Policy evaluation of the MC Control
        q_function = (
            mc_on_policy_step(trajectory, q_function, alpha, gamma)
            if on_policy
            else mc_off_policy_step(
                trajectory,
                q_function,
                # @todo: When I set the eps of the target policy to 0 (Deterministic policy),
                #        the agent will learn nothing and get a full-zero Q function.
                #        I can not figure that out, so I just set the eps to 0.1.
                #        And it learnt successfully, which is what I had not expected.
                epsilon_greedy_policy(n_state, n_action, q_function, eps=0.1),
                policy,
                alpha,
                gamma,
            )
        )

    # After iteration converages, return the optim policy according to the optim Q function
    return q_function, q_function.argmax(axis=1)


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)

    print("\n" + "-" * 25 + "\nBeginning On-policy Monte Carlo control\n" + "-" * 25)
    _, policy_on_mc = learn(True, alpha=0.8, gamma=0.9)
    run_environment(
        env=env,
        policy=Numpy1DArrayPolicy(policy_on_mc),
        path="../../image/mc_on_policy.gif",
        fps=10,
    )

    print("\n" + "-" * 25 + "\nBeginning Off-policy Monte Carlo control\n" + "-" * 25)
    _, policy_off_mc = learn(False, alpha=0.8, gamma=0.9)
    run_environment(
        env=env,
        policy=Numpy1DArrayPolicy(policy_off_mc),
        path="../../image/mc_off_policy.gif",
        fps=10,
    )
