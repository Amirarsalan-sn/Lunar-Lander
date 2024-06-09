# Lunar-Lander
This project demonstrates several soloutions for the LunarLander [environment gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/). The environment is sloved mainly using double dqn and dueling double dqn (d3qn). But, for solving the challenges I faced during the training process, other methods combined with these two methods are tested in order to get the proper result which are fully described in this document. They are combined with different techniques like Boltzman policy, reward wrapping and even an imitation method of learning is used. So bear with me.
## Table of Contents
- [Double DQN](#double-dqn)
  - [Double DQN Simple](#double-dqn-simple)
  - [Double DQN Simple With More Exploration](#double-dqn-simple-with-more-exploration)
  - [Double DQN Boltzman](#double-dqn-boltzman)
- [Dueling Double DQN](#dueling-double-dqn)
  - [Dueling Double DQN With Reward Wrapper](#dueling-double-dqn-with-reward-wrapper)
  - [Hybrid Dueling Double DQN With Reward Wrapper](#hybrid-dueling-double-dqn-with-reward-wrapper)
  - [Simple Hybrid Dueling Double DQN](#simple-hybrid-dueling-double-dqn)
  - [Imitation After D3QN](#imitation-after-d3qn)

## Double DQN
Double DQN is a way of improving the DQN method which seeks to solve the over estimation problem occured in the classical DQN. It also introduces a terget network which is updated periodicaly and helps the main network to reduce its loss easier. In Double DQN, the target value for the main network is computed as deplayed in the image below.

<p align="center">
  <img src="/images/formulas/Screenshot%202024-06-09%20131539.png" alt="Q target in Double DQN">
</p>

### Double DQN Simple
In this section the model is just trained with plain states and rewards of the enviroment (no reward wrapper and state wrapper is applied). Results in episode 1000 of training are as follows:

Episode steps: 1000

Episode reward: 114

<p align="center">
  <img src="images/gif/double_dqn_1000_114.gif" alt="game play">
</p>

### Double DQN Simple With More Exploration
A big problem which I encountered during the training of the last method was the hovering of the lander. I belived that this happened because the time that agent learns to maintatin its stabality, the epsilon was to low so that agent couldn't explore actions which decrease the agent's hight in the stable state. The agent indeed explored actions which decrease the agent's hight but the exploration wasn't from a stable state so most of those decays in hight resulted in crashing which makes the agent think that decrease in hight results in crash and bad rewards. So, I decreased the epsilone decay rate a littl bit to ensure the agent finds the answer starting from a stable state. Results in episode 700 of training are as fallows:

Episode steps: 499

Episode reward: 254

<p align="center">
  <img src="images/gif/double_dqn2_700ep_499st_254r.gif" alt="game play">
</p>

### Double DQN Boltzman
Regarding that the input state space is very big and absolute random exploration might not result in good experiences. It might be a reasonable choice to use whatever the agent has learned in the prior training episodes. It is correct that the agent hasn't learned good things, but, it for sure nows bad the bad actions in the beggining of an episode and that might help us. So I trained the model again using a state wrapper (actually a normilizer of state) and boltzman policy and it improved the performance distinguishably. The results of training are as follows:

**Episode** : 700

Episode steps: 350

Episode reward: 262

<p align="center">
  <img src="images/gif/double_dqn_boltzman_700ep_350st_262r.gif" alt="game play">
</p>

<p align="center">
  <img src="/images/double_boltzman/reward_plot.png" alt="reward" width=350 style="margin-right: 10px;">
  <img src="/images/double_boltzman/Q_value_mean.png" alt="q mean" width=350 style="margin-right: 10px;">
  <img src="/images/double_boltzman/Loss_plot.png" alt="loss" width=350>
</p>

## Dueling Double DQN
Dueling Double DQN(D3QN) is an extension of the Double DQN (Deep Q-Network) algorithm. The key idea behind Dueling Double DQN is to separate the representation of the state value function (V(s)) and the action advantage function (A(s,a)) in the neural network architecture. This allows the model to learn these two components independently, which can lead to better performance in some environments.

<p align="center">
  <img src="images/formulas/DDQN.jpg" alt="architecture">
</p>

### Dueling Double DQN With Reward Wrapper
Dueling Double DQN also suffered from hovering. So, I decided to implement a reward wrapper which prevented the agent from hovering the formula is as follow:

<p align="center">
  <img src="images/formulas/reward_wrap.png" alt="reward wrapper">
</p>

This formula give exponential minus reward to the agent as far as it is hovering. But, hovering is essential for maintaining stabality. So, the agent needs to hover if it wants to gain balance and land safely. Therefore, a Hover Limit is presented into the formula: reward wrapper counts the steps of agents hovering, as far as these steps are below the Hover Limit we are fine. When the hovering steps exceeds the Hover Limit, the agent gets **exponentialy** bad rewards. But, there is another possiblity here, the agent can conclude going up is the best action always but why? Sofar, the agent thought going down would result in failure so it hovers, no we are saying that hovering is also bad so it can conclude going up is the best option available (if the mean of the rewards of "going up" is less than the mean of the reward of "going down"). In order to solve this issue, whenever the agent's speed is higher than a particular amount, a sufficient minus reward is returned. 

The reward wrapper code is as fallows:
```python
    def reward(self, state, reward):
        addition = 0
        if state[3] > 0.55:
            return -10
        if (0.49 < state[3] < 0.52) and (state[1] > 0.52):
            self.bad_counter = min(110, self.bad_counter + 1)
            addition = -math.exp(self.bad_counter - 100)
        else:
            self.bad_counter = 0

        addition = max(-1.5, addition)

        return float(reward) + addition
```
