from train import DistributedTrainer
from test import DistributedTester

import gymnasium as gym
import json



with open('targets.json', 'r') as f:
    targets = json.load(f)
    
print(targets['trains'])



for target in targets['trains']:
    pass

