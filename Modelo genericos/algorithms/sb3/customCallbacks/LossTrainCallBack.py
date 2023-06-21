
from stable_baselines3.common.callbacks import BaseCallback

class LossTrainCallBack(BaseCallback):
    def __init__(self, verbose=0):
        super(LossTrainCallBack, self).__init__(verbose)
        # List used to store total rewards per episode
        self.loss_train = []

    def _on_step(self): 
        long = len(self.locals['infos'])
        for i in range(long):
            if(len(self.locals['infos'][i]) != 0 and 'episode' in self.locals['infos'][i]):
                self.loss_train.append(self.model._logger.name_to_value['train/policy_loss'])

        return True