# Lunar-Lander
This project demonstrates several soloutions for the LunarLander [environment gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/). The environment is sloved mainly using double dqn and dueling double dqn (d3qn). But, for solving the challenges I faced during the training process, other methods combined with these two methods are tested in order to get the proper result which are fully described in this document. They are combined with different techniques like Boltzman policy, reward wrapping and even an imitation method of learning is used. So bear with me.
## Table of Contents
- [Double DQN](##-Double-DQN)
  - [Double DQN Simple](###-Double-DQN-Simple)
  - [Double DQN Boltzman](###-Double-DQN-Boltzman)
- [Dueling Double DQN](##-Dueling-Double-DQN)
  - [Dueling Double DQN With Reward Wrapper](###-Dueling-Double-DQN-With-Reward-Wrapper)
  - [Hybrid Dueling Double DQN With Reward Wrapper](###-Hybrid-Dueling-Double-DQN-With-Reward-Wrapper)
  - [Simple Hybrid Dueling Double DQN](###-Simple-Hybrid-Dueling-Double-DQN)
  - [Imitation After D3QN](###-Imitation-After-D3QN)

## Double DQN
Double DQN is a way of improving the DQN method which seeks to solve the over estimation problem occured in the classical DQN. It also introduces a terget network which is updated periodicaly and helps the main network to reduce its loss easier. In Double DQN, the target value for the main network is computed as deplayed in the image below.
![Q target in Double DQN](/images/formulas/Screenshot%202024-06-09%20131539.png)
