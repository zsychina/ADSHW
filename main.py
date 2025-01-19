import ray
from train import DistributedTrainer
from test import DistributedTester

import gymnasium as gym
import json
import time

import matplotlib.pyplot as plt


with open('targets.json', 'r') as f:
    targets = json.load(f)
    


legends = []
for env in targets.keys():
    for algorithm in targets[env]:

        print(f'Training for {env} - {algorithm}')
        
        trainer = DistributedTrainer(
            env_name=env,
            algo_name=algorithm,
            workers_num=5
        )
        
        trainer.train(episodes=100)
        
        # trainer.algo.save(f'./ckpt/{target["env_name"]}')
        
        print(f'Training for {env} - {algorithm} is done')
        
        time.sleep(5)
        plt.plot(trainer.episode_return_mean())
        legends.append(f'{algorithm}')
        
        ray.shutdown() 


    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.legend(legends)
    plt.savefig(f'./assets/{env}.png')
    
    plt.clf()
