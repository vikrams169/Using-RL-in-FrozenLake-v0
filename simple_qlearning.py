#This python file aims at using a basic Q-Learning implementation to train an agent in the FrozenLake-v0 environment of gym

import numpy as np
import matplotlib.pyplot as plt
import gym

#Creating a class for the agent

class Agent(object):

	#Initializing the agent's parameters
	def __init__(self,env,gamma=0.98,alpha=0.7):

		self.env=env
		self.state_size=env.observation_space.n							#Number of possible states/configurations 
		self.action_size=env.action_space.n								#Number of possible actions to take 
		self.gamma=gamma												#Discount factor to take future rewards into acount
		self.alpha=alpha												#Learning rate, to update the Q-table
		self.action_space=np.arange(self.action_size)					#Defining the array of the possible actions the agent can take
		self.qtable=np.zeros((self.state_size,self.action_size))		#Initializing the Q-Table
		self.reward_history=[]											#To maintain track of average reward per episode while training
		self.episode_lengths=[]											#To keep track of the length od episodes

	#Using the epsilon greedy policy to choose the most appropriate action from the action space
	def epsilon_greedy(self,state,epsilon=0.2):

		qvalues=self.qtable[state,:]
		A=np.zeros((self.action_size)) + epsilon/self.action_size
		greedy_action=np.argmax(qvalues)
		A[greedy_action]+=1-epsilon
		return np.random.choice(self.action_space,p=A)

	#Updating the qtable for each new state vsited (for the first time) in each episode
	def update_qtable(self,state,action,reward,next_state):

		next_action=int(np.argmax(self.qtable[next_state,:]))
		self.qtable[state,action]+=self.alpha*(reward+self.gamma*self.qtable[next_state,next_action]-self.qtable[state,action])

	#Defining how a single episode would roll out
	def qlearning_episode(self):

		#Initialising the episode buffer to store the current state, chosen action, and reward at that particular timestep 
		episode=[]
		#Starting from the initial starting point for the beginning of each episode
		state_now=self.env.reset()
		while True:
			#Choosing the action as per the constant epsioln greedy policy
			action=self.epsilon_greedy(state_now)
			next_state,reward,done,_=self.env.step(action)
			episode.append([state_now,action,reward,next_state])
			state_now=next_state
			#If the goal is reached, terminate the episode, otherwise go to the next state
			if done==True:
				break
		#Returning the buffer containing the information for each timestep of the episode
		return np.array(episode)

	#Training the agent through a series of episodes
	def train(self,num_episodes=10000):

		#Iterating over a number of episodes
		for i in range(num_episodes):
			#Generating the episode
			episode=self.qlearning_episode()
			#Keeping track of the length of the episode
			self.episode_lengths.append(len(episode[:,0]))
			#Updating the particular qvalues only
			for k in range(len(episode[:,0])):
				self.update_qtable(int(episode[k,0]),int(episode[k,1]),float(episode[k,2]),int(episode[k,3]))
			self.reward_history.append(np.mean(episode[:,2]))
			if (i+1)%100==0:
				print("Average reward claimed by the agent in episode {} : {}".format(i+1,self.reward_history[-1]))
				print("Length of episode {} : {}".format(i+1,self.episode_lengths[-1]))


#Choosing the environment (though this code should for any discrete/non continuous state environment)
#For a different environment, the state size and action size should be chosen accordingly
env=gym.make("FrozenLake-v0")

agent=Agent(env=env)
agent.train()

reward_history=agent.reward_history

#Plotting the averge rewards and episode Lengths gained throughout each episode per episode 
fig, axs = plt.subplots(1,2)
axs[0].plot(agent.reward_history)
axs[0].set_title('Average Reward per Episode')
axs[1].plot(agent.episode_lengths, 'tab:orange')
axs[1].set_title('Episode_Length')

plt.show()

print('Average Reward throughout all episodes={}'.format(sum(reward_history)/len(reward_history)))