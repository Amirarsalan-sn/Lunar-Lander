# Lunar-Lander
This project demonstrates several soloutions for the LunarLander [environment gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/). The environment is sloved mainly using double dqn and dueling double dqn (d3qn). But, for solving the challenges I faced during the training process, other methods combined with these two methods are tested in order to get the proper result which are fully described in this document. They are combined with different techniques like Boltzman policy, reward wrapping and even an imitation method of learning is used. So bear with me.
## Table of Contents
- [Double DQN](##-double-dqn)
  - [Double DQN Simple](###-double-dqn-simple)
  - [Double DQN Simple With More Exploration](###-double-dqn-simple-with-more-exploration)
  - [Double DQN Boltzman](###-double-ddn-boltzman)
- [Dueling Double DQN](##-dueling-double-dqn)
  - [Dueling Double DQN With Reward Wrapper](###-dueling-double-dqn-with-reward-wrapper)
  - [Hybrid Dueling Double DQN With Reward Wrapper](###-hybrid-dueling-double-dqn-with-reward-wrapper)
  - [Simple Hybrid Dueling Double DQN](###-simple-hybrid-dueling-double-dqn)
  - [Imitation After D3QN](###-imitation-after-d3qn)

## Double DQN
Double DQN is a way of improving the DQN method which seeks to solve the over estimation problem occured in the classical DQN. It also introduces a terget network which is updated periodicaly and helps the main network to reduce its loss easier. In Double DQN, the target value for the main network is computed as deplayed in the image below.

<p align="center">
  <img src="/images/formulas/Screenshot%202024-06-09%20131539.png" alt="Q target in Double DQN">
</p>

### Double DQN Simple
In this section the model is just trained with plain states and rewards of the enviroment (no reward wrapper and state wrapper is applied). Results in episode 1000 of training are as follows:

Episode steps: 1000

Episode reward: 114

![game play](/images/gif/double_dqn_1000_114.gif)

### Double DQN Simple With More Exploration
A big problem which I encountered during the training of the last method was the hovering of the lander. I belived that this happened because the time that agent learns to maintatin its stabality, the epsilon was to low so that agent couldn't explore actions which decrease the agent's hight in the stable state. The agent indeed explored actions which decrease the agent's hight but the exploration wasn't from a stable state so most of those decays in hight resulted in crashing which makes the agent think that decrease in hight results in crash and bad rewards. So, I decreased the epsilone decay rate a littl bit to ensure the agent finds the answer starting from a stable state. Results in episode 700 of training are as fallows:

Episode steps: 499

Episode reward: 254

![game play](/image/gif/double_dqn2_700ep_499st_254r.gif)

### Double DQN Boltzman

