#This python file aims at using a basic monte carlo implementation to train an agent in the FrozenLake-v0 environment of gym

import numpy as np
import matplotlib.pyplot as plt
import gym

#Creating a class for the agent

class Agent(object):

	#Initializing the agent's parameters
	def __init__(self,state_size,action_size,gamma=1,alpha=0.5):

		self.state_size=state_size										#Number of possible states/configurations 
		self.action_size=action_size									#Number of possible actions to take 
		self.gamma=gamma												#Discount factor to take future rewards into acount
		self.alpha=alpha												#Learning rate, to update the Q-table
		self.action_space=np.arange(self.action_size)					#Defining the array of the possible actions the agent can take
		self.qtable=np.zeros((self.state_size,self.action_size))		#Initializing the Q-Table

	#Using the epsilon greedy policy to choose the most appropriate action from the action space
	def epsilon_greedy(self,state,epsilon=0.2):

		qvalues=self.qtable[state,:]
		A=np.zeros((self.action_size)) + epsilon/self.action_size
		greedy_action=np.argmax(qvalues)
		A[greedy_action]+=1-epsilon
		return np.random.choice(self.action_space,p=A)

	#Defining a function to discount rewards of an entire episode
	def discounted_rewards(self,rewards):

		#Rewards is a 1-D array with the stored rewards obtained at each timestep of an episode
		current_reward=0
		discounted_rewards=np.zeros((len(rewards)))
		for t in reversed(range(len(rewards))):
			current_reward = self.gamma*current_reward + rewards[t]
			discounted_rewards[t]=current_reward
		return discounted_rewards

	#Updating the qtable for each new state vsited (for the first time) in each episode
	def update_qtable(self,state,action,reward):

		self.qtable[state,action]+=self.alpha*(reward-self.qtable[state,action])


#Choosing the environment (though this code should for any discrete/non continuous state environment)
#For a different environment, the state size and action size should be chosen accordingly
env=gym.make("FrozenLake-v0")

state_size=env.observation_space.n
#For the FrozenLake-v0 environment, this turns out to be 16
action_size=env.action_space.n
#For the FrozenLake-v0 environment, this turns out to be 4

#Defining how a single episode would roll out
def monte_carlo_episode(agent):

	#Initialising the episode buffer to store the current state, chosen action, and reward at that particular timestep 
	episode=[]
	#Starting from the initial starting point for the beginning of each episode
	state_now=env.reset()
	while True:
		#Choosing the action as per the constant epsioln greedy policy
		action=agent.epsilon_greedy(state_now)
		state_next,reward,done,_=env.step(action)
		episode.append([state_now,action,reward])
		state_now=state_next
		#If the goal is reached, terminate the episode, otherwise go to the next state
		if done==True:
			break
	#Returning the buffer containing the information for each timestep of the episode
	return np.array(episode)

#Training the agent through a series of episodes
def training(agent,num_episodes=3000):

	#Maintaining a buffer to store average reward for every episode
	reward_history=[]
	#Iterating over a number of episodes
	for i in range(num_episodes):
		#Generating the episode
		episode=monte_carlo_episode(agent)
		#Storing the reards for the episode, and discounting it
		rewards=episode[:,2]
		rewards=agent.discounted_rewards(rewards)
		#Using a first visit monte carlo approach
		#Thus the qvalues of only the first visits to a particular state are updated
		visited_states=[]
		visited_actions=[]
		visited_rewards=[]
		#Obtaining the indexes of the first visits to states in the episode
		for j in range(len(episode[:,0])):
			if episode[j,0] not in visited_states:
				visited_states.append(episode[j,0])
				visited_actions.append(episode[j,1])
				visited_rewards.append(rewards[j])
		#Updating those particular qvalues only
		for k in range(len(visited_states)):
			agent.update_qtable(int(visited_states[k]),int(visited_actions[k]),float(visited_rewards[k]))
		reward_history.append(episode[0,2])
		if i%100==0:
			print("Average reward claimed by the agent in episode {} : {}".format(i+1,reward_history[-1]))
	return reward_history

agent=Agent(state_size=state_size,action_size=action_size)

reward_history=training(agent)

#Plotting the averge rewards gained throughout each episode per episode 
'''plt.plot(reward_history)
plt.title("Average Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.show()'''





