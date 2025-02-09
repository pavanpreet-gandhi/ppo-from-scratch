import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from network import FeedForwardNN

class PPO:
    
    def __init__(self, env):
        """
        Initializes the PPO agent.
        
        Args:
            env: the environment to interact with
        """
        # extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        # initialize hyperparameters
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.cov_mat = 0.5 * torch.eye(self.act_dim) # covariance matrix for action distribution NOTE: this can alse be learned to allow for adaptive exploration and entropy regularization
        self.gamma = 0.95 # discount factor
        self.n_epochs_per_iteration = 5 # number of epochs of optimization of the actor per iteration NOTE: this can be adjusted based on KL divergence between the old and new policies or the size of policy improvement
        self.clip = 0.2 # clip parameter for PPO
        self.lr = 0.005 # same learning rate for both actor and critic
        
        # initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        
        # itialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
    
    
    def get_action(self, obs):
        """
        Gets a (noisy) action from the actor network given an observation. The noise comes from a pre-defined covariance matrix (see hyperparameters).
        
        Args:
            obs: the current observation
        
        Returns:
            action: the action to take
            log_prob: the log probability of the action
        """
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach() # detach because we don't need to keep track of any gradients
    
    
    def compute_rewards_to_go(self, batch_rewards):
        """
        Compute the rewards-to-go for each timestep in a batch of episodes given the rewards for each episode.
        These are effectively sampled Q-values for each state-action pair.
        
        Args:
            batch_rewards: a list of lists, with the inner list containing the rewards for each timestep of an episode
        
        Returns:
            rewards_to_go: a tensor containing the rewards-to-go for each timestep in the batch
        """
        batch_rewards_to_go = []
        for episode_rewards in reversed(batch_rewards):
            rtg = 0
            for r in reversed(episode_rewards):
                rtg = r + self.gamma * rtg
                batch_rewards_to_go.insert(0, rtg)
        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype=torch.float)
        return batch_rewards_to_go


    def rollout(self):
        """
        Roll out a batch of episodes to collect `timestep_per_batch` timesteps of data.
        
        Returns:
            batch_obs: a tensor of shape (timesteps_per_batch, obs_dim)
            batch_acts: a tensor of shape (timesteps_per_batch, act_dim)
            batch_log_probs: a tensor of shape (timesteps_per_batch)
            batch_rewards_to_go: a tensor of shape (timesteps_per_batch)
            batch_lens: a tensor of shape (number of episodes)
        """
        # batch arrays to hold data from rollout
        batch_obs = [] # (timesteps_per_batch, obs_dim)
        batch_acts = [] # (timesteps_per_batch, act_dim)
        batch_log_probs = [] # (timesteps_per_batch)
        batch_rewards_to_go = [] # (timesteps_per_batch)
        
        # eposide data
        # NOTE: the the number of timesteps per episode is not fixed, this is an intermediate list of lists used to compute rewards_to_go
        batch_rewards = [] # (number of episodes, number of timesteps per episode)
        batch_lens = [] # (number of episodes)
        
        t_batch = 0 # number of timesteps run so far in this batch
        while t_batch < self.timesteps_per_batch:
            
            # reset episode-specific variables
            episode_rewards = [] # holds the rewards for the current episode
            obs, _ = self.env.reset()
            done = False
            
            for t_ep in range(self.max_timesteps_per_episode): # play through an episode
                
                # break if we've collected enough timesteps
                if t_batch >= self.timesteps_per_batch:
                    break
                
                # take action and store data (obs, action, log_prob, reward (within episode))
                t_batch += 1
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, reward, done, _, _ = self.env.step(action)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                episode_rewards.append(reward)
                
                if done: # end episode if done
                    break
            
            # append episode data if we collected any timesteps
            if t_ep > 0:
                batch_lens.append(t_ep)
                batch_rewards.append(episode_rewards)
        
        # convert to tensors
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)
        
        return batch_obs, batch_acts, batch_log_probs, batch_rewards_to_go, batch_lens
                
    
    def evaluate(self, batch_obs, batch_acts):
        """
        Evaluates the value function of a batch of observations according to the critic and the log probability of the actions taken in the batch according to the actor.
        
        Args:
            batch_obs: a tensor of shape (timesteps_per_batch, obs_dim)
            batch_acts: a tensor of shape (timesteps_per_batch, act_dim)
        
        Returns:
            V: a tensor of shape (timesteps_per_batch) corresponding to the value function of each observation according to the critic
            log_probs: a tensor of shape (timesteps_per_batch) corresponding to the log probability of each action taken in the batch according to the actor
        """
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs
    
    
    def learn(self, total_timesteps):
        """
        Trains the actor and critic networks using PPO.
        
        Args:
            total_timesteps: the total number of timesteps to train for
        
        Returns:
            None
        """
        t_so_far = 0 # timesteps simulated so far
        while t_so_far < total_timesteps:
            
            # collect a batch of data from rollout
            batch_obs, batch_acts, batch_log_probs, batch_rewards_to_go, batch_lens = self.rollout()
            # NOTE: each entry in rewards_to_go is a sample of the Q-function at the corresponding state-action pair
            
            # compute advantages
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rewards_to_go - V.detach() # advantages associated with each state-action pair in the batch, shape (timesteps_per_batch)
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # normalize advantages NOTE: these are fixed before updating the actor and critic
            
            for epoch in range(self.n_epochs_per_iteration):
                
                # calculate V(s_t) and pi_theta(a_t | s_t) for all timesteps in the batch to compute ratios
                V, current_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(current_log_probs - batch_log_probs)
                
                # calculate loss
                surr_1 = ratios * A_k
                surr_2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                actor_loss = -torch.min(surr_1, surr_2).mean() # i.e. we want to maximize the minimum of the two surrogate losses
                
                # optimize actor
                # NOTE: the thing that changes in each epoch is the current_log_probs of the actions taken in the batch, which affects the ratios
                # we want to push up the probabilities of good actions in the actor relative to the probability of taking them before optimization (i.e. batch_log_probs)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # optimize critic
                # NOTE: the critic simply tries to minimize the MSE between the rewards-to-go and the value function
                critic_loss = F.mse_loss(V, batch_rewards_to_go)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
            
            t_so_far += sum(batch_lens)


if __name__ == "__main__":
    import gymnasium as gym
    env = gym.make("Pendulum-v1")
    model = PPO(env)
    model.learn(100000)