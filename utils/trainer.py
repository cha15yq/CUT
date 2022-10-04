import os
import logging
from datetime import datetime
from utils.logger import logger


class Trainer(object):
    def __init__(self, args):
        self.args = args
        sub_dir = args.content
        self.save_dir = os.path.join(args.save_dir, sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        logger(os.path.join(self.save_dir, 'train.log'))
        for k, v in args.__dict__.items():
            logging.info("{}:{}".format(k, v))

    def setup(self):
        pass

    def train(self):
        pass
