import os

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

import gymnasium as gym

class DistributedTester:
    def __init__(self, env_name, workers_num=1):
        self.env_name = env_name
        self.workers_num = workers_num
        self.config = None
        
        register_env('my' + self.env_name, self.env_creator)
        ray.init()
        
        self.algo = Algorithm.from_checkpoint(path=os.path.abspath('./ckpt'))


    def env_creator(self, env_config):
        return gym.make(self.env_name)


    def run_inference(self, num_episodes=10):
        # Create the environment
        env = self.env_creator(None)
        
        # Run inference for the specified number of episodes
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Get action from the algorithm
                action = self.algo.compute_single_action(obs)
                
                # Take the action in the environment
                obs, reward, done, info = env.step(action)
                
                # Accumulate the reward
                total_reward += reward
                
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    
    


if __name__ == '__main__':
    test = DistributedTester(
        env_name='CartPole-v1',
        workers_num=1
    )
    
    print(test.algo)
    
    
    test.run_inference(num_episodes=10)