import os

import ray
from ray.tune.registry import register_env

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.impala import IMPALAConfig
from ray.rllib.algorithms.appo import APPOConfig

import gymnasium as gym



class DistributedTrainer:
    def __init__(self, env_name, algo_name='PPO', workers_num=1):
        self.algo_name = algo_name
        self.env_name = env_name
        self.workers_num = workers_num
        self.config = None
        self.train_result = []
        
        register_env('my' + self.env_name, self.env_creator)
        ray.init()
        
        if self.algo_name == 'PPO':
            self.config = (
                PPOConfig()
                .environment(env='my' + self.env_name)
                .env_runners(num_env_runners=self.workers_num)
                .training(
                    gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size_per_learner=256
                )
            )
            
            
        elif self.algo_name == 'DQN':
            self.config = (
                DQNConfig()
                .environment(env='my' + self.env_name)
                .training(replay_buffer_config={
                    "type": "PrioritizedEpisodeReplayBuffer",
                    "capacity": 60000,
                    "alpha": 0.5,
                    "beta": 0.5,
                })
                .env_runners(num_env_runners=self.workers_num)
            )
            
        elif self.algo_name == 'IMPALA':
            self.config = (
                IMPALAConfig()
                .environment(env='my' + self.env_name)
                .env_runners(num_env_runners=self.workers_num)
                .training(lr=0.0003, train_batch_size_per_learner=512)
                .learners(num_learners=1)
            )
            
        elif self.algo_name == 'APPO':
            self.config = (
                APPOConfig()
                .training(lr=0.01, grad_clip=30.0, train_batch_size_per_learner=50)
                .environment(env='my' + self.env_name)
                .env_runners(num_env_runners=self.workers_num)
                .learners(num_learners=1)
            )
        

        if self.config is None:
            raise ValueError('Invalid algorithm name')
        
        self.algo = self.config.build()

    def env_creator(self, env_config):
        return gym.make(self.env_name)

    
    def train(self, episodes=50):
        for episode_i in range(episodes):
            print(f'Episode {episode_i}')
            self.train_result.append(self.algo.train())
            checkpoint_dir = os.path.abspath('./ckpt')
            save_result = self.algo.save(checkpoint_dir)
            path_to_checkpoint = save_result.checkpoint.path
            print(f'{path_to_checkpoint=}')
            
            self.nodes, self.actors = self.check_worker_status()
            self.handle_worker_failure(self.nodes, self.actors)
        

    def check_worker_status(self):
        nodes = ray.nodes()
        for node in nodes:
            print(f"Node {node['NodeID']} - {node['Alive']}")
        # print(nodes)

        actors = ray.state.actors()
        for actor_id, actor_data in actors.items():
            print(f"Actor {actor_id} - {actor_data['State']}")
        # print(actors)
        return nodes, actors
    
    
    
    def handle_worker_failure(self, nodes, actors):
        for node in nodes:
            if not node['Alive']:
                print(f"Node {node['NodeID']} is dead. Restarting Ray...")
                ray.shutdown()
                # 添加延迟以避免频繁重启
                time.sleep(5)
                ray.init()
                break
            
        for actor_id, actor_data in actors.items():
            if actor_data['State'] == 'DEAD':
                print(f"Actor {actor_id} is dead. Killing actor...")
                try:
                    ray.kill(actor_id)
                except Exception as e:
                    print(f"Failed to kill actor {actor_id}: {e}")
                break



    def episode_return_mean(self):
        ep_return_mean = []
        for result in self.train_result:
            try:
                ep_return_mean.append(result['env_runners']['episode_return_mean'])
            except KeyError:
                pass    
                
        return ep_return_mean



if __name__=='__main__':
    trainer = DistributedTrainer(env_name='CartPole-v1', 
                                 algo_name='PPO', 
                                 workers_num=3)
    
    trainer.train(episodes=5)

    # 确保以上执行完毕
    
    import time
    time.sleep(5)
    
    import matplotlib.pyplot as plt
    plt.plot(trainer.episode_return_mean())
    plt.show()

    ray.shutdown()  
    
    