# Using-RL-in-FrozenLake-v0
This repository displays the use of Reinforcement Learning, particularly Q-Learning and Monte Carlo methods to play the FrozenLake-v0 Environment of OpenAI Gym.  
<br>
The Frozen Lake environment can be better explained or reviwed by going to the souce code <a href="https://github.com/openai/gym/wiki/FrozenLake-v0">here</a>.
<br>In this environment, there exists a 4x4 gridworld, where each square represents either a safe spot (stepping on which gives 0 reward), hole (stepping on which completes the episode, with 0 reward), and the final goal (staepping on which successfully finishes the episode with a reward of 1).
<br>
<br>
The agent can move either north, south, east, or west, except at the edged/corners of the world. There thus exists a discrete state space of size 16, and also a discrete action space of size 4.
<br>
<br>
This environment has been solved with the objective of reaching maximum reward (thus reaching the final goal), and has been done so, by using three reinforcement learning techniques, each trained on 10,000 episodes.
<br>
<br>
To better play this environment, there are three reinforcement learning techniques used, and compared:
<br>
<h3>1. Simple Every Vist Monte Carlo Method</h2>
<br>No bootstrapping, and updates done at the end of every episode, using a maintained Q-Table.
<br>The average rewards and episode lengths look like:
<br><center><img src="simple_montecarlo.png"></center>
<br>
<h3>2. Simple QLearning Method</h2>
<br>Uses bootstrapping, and updates done at the each timestep of every episode, using a maintained Q-Table.
<br>The average rewards and episode lengths look like:
<br><center><img src="simple_qlearning.png"></center>
<br>
<h3>3. Deep QLearning Method</h2>
<br>Uses bootstrapping, and updates done at the each timestep of every episode, using a neural network function approximator.
<br>The average rewards and episode lengths look like:
<br><center><img src="deep_qlearning.png"></center>
