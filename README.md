# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description



FrozenLake-v1 is a grid-based reinforcement learning environment provided by Gymnasium. The environment contains frozen tiles, holes, a starting state, and a goal state. The agent starts from the starting position and must learn to reach the goal while avoiding the holes. At each state, the agent can choose one of four actions: Left, Down, Right, or Up. The agent receives a reward when it successfully reaches the goal. The environment is used to train the agent through repeated episodes and learn the best action for each state using Monte Carlo Control.




## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm

1.Initialize the FrozenLake environment and obtain the number of states and actions.

2.Initialize the Q-table with zeros.

3.Set the hyperparameters: number of episodes, learning rate α, discount factor γ, and epsilon values.

4.Select actions using the epsilon-greedy policy.

5.Generate a complete episode until the agent reaches the goal, falls into a hole, or reaches the maximum number of steps.

6.Calculate the return Gt by traversing the episode backward.

7.Update the Q-value using: Q(s,a) ← Q(s,a) + α[Gt − Q(s,a)]

8.Gradually reduce epsilon to shift from exploration to exploitation.

9.Extract the greedy policy using the maximum Q-value for each state.

10.Display the Q-table, state-value function, learned policy, average reward, and learning curve.

## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
def epsilon_greedy_action(state, epsilon):
    # Exploration
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    # Exploitation
    return np.argmax(Q[state])

…    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Display progress
    if (episode + 1) % 2000 == 0:
        print(
            "Episode:", episode + 1,
            "Average Reward:",
            np.mean(episode_rewards[-1000:])
        )



```

---

## Output

### Number of Episodes : 20000

<img width="938" height="589" alt="image" src="https://github.com/user-attachments/assets/616db978-c2a0-4e7f-85cb-90bce402b6e2" />

<img width="836" height="482" alt="image" src="https://github.com/user-attachments/assets/2d94e9e2-5276-4aae-b2b5-a03ab009cf2e" />



### Number of Episodes : 2000
<img width="507" height="571" alt="image" src="https://github.com/user-attachments/assets/e296be20-a569-4986-acf3-f7b762acf1fe" />


<img width="869" height="471" alt="image" src="https://github.com/user-attachments/assets/76cb7dc9-0ab5-4da9-a71e-11516237cfdb" />

---

## Result


The On-Policy Monte Carlo Control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment.The agent learned the action-value function Q(s,a) from complete episodes and obtained an improved policy using epsilon-greedy action selection.



## Inference

The experiment demonstrates how an agent can learn the best actions by interacting with the environment. Monte Carlo Control updates the Q-values based on the rewards obtained after completing each episode. The epsilon-greedy method helps the agent explore new actions while also selecting actions that have already shown good results. After many episodes, the agent learns an effective policy to reach the goal and avoid dangerous states.






---

