# coding: utf-8
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, ".."))

from utils.policy_wrapper import Numpy1DArrayPolicy
from utils.run_gym import run_environment

register(
    id="SlipperyFrozenLake-v1",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": True},
)
register(
    id="FrozenLake-v1",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": False},
)


def policy_evaluation(transition, n_state, n_action, policy, gamma=0.9, eps=1e-3):
    """
    Policy evaluation (PE):

    Use Bellman's equation to iterate until converaged to get the value function of current policy.
    """

    value_function = np.zeros(n_state)
    prev_value_function = value_function
    while True:
        next_value_function_raw = []
        for state in range(n_state):
            action = policy[state]
            value = 0
            for next_state in range(n_state):
                filtered_pair = [
                    pair for pair in transition[state][action] if pair[1] == next_state
                ]
                if len(filtered_pair) == 0:
                    continue

                reward = filtered_pair[0][2]
                probability = filtered_pair[0][0]
                value += probability * (reward + gamma * value_function[next_state])

            next_value_function_raw.append(value)

        prev_value_function = value_function
        value_function = np.array(next_value_function_raw)

        if all(abs(prev_value_function - value_function) < eps):
            break

    return value_function


def policy_improvement(
    transition, n_state, n_action, value_from_policy, policy, gamma=0.9
):
    """
    Policy improvement (PM):

    For every state, choose the action that will bring the highest reward (Q value function).
    """

    new_policy = np.zeros(n_state, dtype=int)
    for state in range(n_state):
        max_value = -1
        action_buf = -1
        for action in range(n_action):
            value = 0
            for next_state in range(n_state):
                filtered_pair = [
                    pair for pair in transition[state][action] if pair[1] == next_state
                ]
                if len(filtered_pair) == 0:
                    continue

                reward = filtered_pair[0][2]
                probability = filtered_pair[0][0]
                value += probability * (reward + gamma * value_from_policy[next_state])

            if value > max_value:
                max_value = value
                action_buf = action

        new_policy[state] = action_buf

    return new_policy


def policy_iteration(transition, n_state, n_action, gamma=0.9, eps=10e-3):
    """
    Policy iteration (PI):

    Run policy evaluation (PE) & policy improvement (PM) alternatively until converged.

    Policy evaluation uses Bellman's equation (not the optim one) to get the value function of current policy.

    Policy improvement greedily generates a newer and better policy by the value function from PE.

    The final policy is the optim policy.

    PI needs full knowledge of the environment (mainly the transition function `P(s' | s, a)`).
    """

    value_function = np.zeros(n_state)
    policy = np.zeros(n_state, dtype=int)

    prev_evaluation = None
    while True:
        evaluation = policy_evaluation(
            transition, n_state, n_action, policy, gamma, eps
        )
        policy = policy_improvement(
            transition, n_state, n_action, evaluation, policy, gamma
        )

        if prev_evaluation is None or not all(abs(prev_evaluation - evaluation) < eps):
            prev_evaluation = evaluation
        else:
            value_function = evaluation
            break

    return value_function, policy


def value_iteration(transition, n_state, n_action, gamma=0.9, eps=1e-3):
    """
    Value iteration (VI):

    Use optim Bellman's equation to iterate until converged to get the optim value function.

    VI needs full knowledge of the environment (mainly the transition function `P(s' | s, a)`).
    """

    # Iterate the value function
    value_function = np.zeros(n_state)
    prev_value_function = value_function
    while True:
        next_value_function_raw = []
        for state in range(n_state):
            max_value = -1
            for action in range(n_action):
                value = 0
                for next_state in range(n_state):
                    filtered_pair = [
                        pair
                        for pair in transition[state][action]
                        if pair[1] == next_state
                    ]
                    if len(filtered_pair) == 0:
                        continue

                    reward = filtered_pair[0][2]
                    probability = filtered_pair[0][0]
                    value += probability * (reward + gamma * value_function[next_state])

                if value > max_value:
                    max_value = value

            next_value_function_raw.append(max_value)

        prev_value_function = value_function
        value_function = np.array(next_value_function_raw)

        if all(abs(prev_value_function - value_function) < eps):
            break

    # Generate optim policy by optim value function
    policy = np.zeros(n_state, dtype=int)
    for state in range(n_state):
        max_value = -1
        action_buf = -1
        for action in range(n_action):
            value = 0
            for next_state in range(n_state):
                filtered_pair = [
                    pair for pair in transition[state][action] if pair[1] == next_state
                ]
                if len(filtered_pair) == 0:
                    continue

                reward = filtered_pair[0][2]
                probability = filtered_pair[0][0]
                value += probability * (reward + gamma * value_function[next_state])

            if value > max_value:
                max_value = value
                action_buf = action

        policy[state] = action_buf

    return value_function, policy


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="rgb_array")
    # env = gym.make("SlipperyFrozenLake-v1", render_mode="rgb_array")

    env = TimeLimit(env, max_episode_steps=100)
    transition = env.unwrapped.P
    n_state = env.observation_space.n
    n_action = env.action_space.n

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)
    value_function_pi, policy_pi = policy_iteration(
        transition, n_state, n_action, gamma=0.9, eps=1e-3
    )
    run_environment(
        env=env,
        policy=Numpy1DArrayPolicy(policy_pi),
        path="../../image/dp_pi.gif",
        fps=10,
    )

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
    value_function_vi, policy_vi = value_iteration(
        transition, n_state, n_action, gamma=0.9, eps=1e-3
    )
    run_environment(
        env=env,
        policy=Numpy1DArrayPolicy(policy_vi),
        path="../../image/dp_vi.gif",
        fps=10,
    )
