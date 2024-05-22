import torch
import glob
import re
from pathlib import Path
import logging


class EarlyStopper(object):
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

class HardEarlyStopper(object):
    def __init__(self, num_trials, save_path, path_exist):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = self.__path_increment(save_path, path_exist)

    def is_continuable(self, model, accuracy, threshold=0.1):
        logging.basicConfig(format='(%(asctime)s) - [%(levelname)s] : %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        logger = logging.getLogger(__name__)

        if self.best_accuracy == 0:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.pct_increase(accuracy) >= threshold:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            logger.info(f'Early stopping, trial : {self.trial_counter}/{self.num_trials}')
            return True
        else:
            logger.info(f'Early stopping, trial : {self.trial_counter}/{self.num_trials}')
            return False

    def __path_increment(self, path: str, path_exist: bool, sep=''):
        """
        i.e. runs/exp => runs/exp{sep}0, runs/exp{sep}1 etc.
        """
        path = Path(path)  # os-agnostic
        if (path.exists() and path_exist) or (not path.exists()):
            return str(path)
        else:
            dirs = glob.glob(f"{path}{sep}*")  # similar paths
            matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]  # indices
            n = max(i) + 1 if i else 2  # increment number
            return f"{path}{sep}{n}"  # update path

    def pct_increase(self, accuracy):
        """
        pct increase: (A-B)/abs(B) * 100
        """
        pct = (accuracy - self.best_accuracy) / abs(self.best_accuracy) * 100
        return pct
