# Collaboration and Competition - README

---

# Introduction

This assignment performs multi-agent reinforcement learning on the Unity **Tennis** envitonment. In this environment two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of `+0.1`.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of `-0.01`. 

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

In order to solve the environment, the agents must get an average score of `+0.5` over 100 consecutive episodes, after taking the maximum over both agents.


# Instructions

1. `git clone` this repo: https://github.com/bohana/udacity-deep-rl.git
1. `pip install` the following packages: `numpy`, `matplotlib`, `pandas`, `pytorch`, `unityagents (0.4.0)`
1. Download the Unity `Tennis` environment per project instructions (I used the Linux version found [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)), and unzip it in the repo's `p3_collab-compet` directory.
1. Run this notebook.

## Project Files

This repo contains the following files under the `p2_control` directory:

```
P3_Collab_Compete_Tennis.ipynb  - notebook with entire project implementation
checkpoint_ag0_actor.pth        - actor network params - agent 1 
checkpoint_ag0_critic.pth       - cricit network params - agent 1 
checkpoint_ag1_actor.pth        - actor network params - agent 2
checkpoint_ag1_critic.pth       - critic network params - agent 2
README.md                       - this file.
```

## Training the agent

* Just run the notebook `P3_Collab_Compete_Tennis.ipynb` all the way to the end. The `pth` files will be generated once training completes. 

# References

[1] - Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (Lowe et al 2017) - https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf

[2] - Continous Control with Deep Reinforcement Learning (Lilicrap et al 2015) https://arxiv.org/pdf/1509.02971.pdf

[3] - Benchmarking Deep Reinforcement Learning for Continuous Control (Duan et al 2016) https://arxiv.org/pdf/1604.06778.pdf

[4] - https://towardsdatascience.com/training-two-agents-to-play-tennis-8285ebfaec5f
