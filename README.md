# Using-RL-in-FrozenLake-v0
This repository displays the use of Reinforcement Learning, particularly Q-Learning and Monte Carlo methods to play the FrozenLake-v0 Environment of OpenAI Gym.  
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
