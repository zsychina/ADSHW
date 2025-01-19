import ray

from train import DistributedTrainer


class DistributedTester(DistributedTrainer):
    def __init__(self, env_name, algo_name='PPO', workers_num=1):
        super().__init__(env_name, algo_name='PPO', workers_num=1)
        
        
    def test(self, episodes=50):
        pass
        


