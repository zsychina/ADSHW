from train import DistributedTrainer
from test import DistributedTester

import gymnasium as gym
import json

import matplotlib.pyplot as plt


with open('targets.json', 'r') as f:
    targets = json.load(f)
    


for target in targets['trains']:
    trainer = DistributedTrainer(
        env_name=target['environment'],
        algo_name=target['algorithm'],
        workers_num=5
    )
    
    trainer.train(episodes=5)
    
    # trainer.algo.save(f'./ckpt/{target["env_name"]}')
    
    print(f'Training for {target["environment"]} is done')
    
    plt.plot(trainer.episode_return_mean())
    plt.savefig(f'./assets/{target["environment"]}.png')
    
    
    del trainer

