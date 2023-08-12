# Dynamic Programming (DP)

When we have full knowledge of the environment, which means we completely know the transition function $\mathbb{P}_{\mathcal E}(s' \mid s, a)$, we can use DP to get the optim policy.

According to the optim Bellman's equation, the value function of the optim policy should also be optim and satisfy:

$$
{\color{red} V^\star(s)} = \max_{a \in \mathcal{A}} \left[\sum_{s' \in \S} \mathbb{P}_{\mathcal E}(s' \mid s, a)[r(s, a, s') + \gamma {\color{red} V^\star(s')}]\right]
$$

We can use iteration to solve this equation to get the optim value function. This leads to **value iteration (VI)**.

VI iterates the value function. On the other hand, PI iterates policy itself.

PI starts at a default policy and runs following steps alternatively until it converges:

- (Policy evaluation) Get the value function $V^\pi$ of current policy $\pi$
- (Policy improvement) Generate a better policy $\pi'$ according to the value function $V^\pi$

VI & PI are all DP algorithms.
