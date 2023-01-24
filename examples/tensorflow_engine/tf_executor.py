# -*- coding: utf-8 -*-

import os
import sys

import tensorflow as tf
from tf_client import Customized_Client

import fedscale.cloud.config_parser as parser
from fedscale.cloud.execution.executor import Executor
from fedscale.cloud.logger.execution import *

"""In this example, we only need to change the Client Component we need to import"""

class Customized_Executor(Executor):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""

    def __init__(self, args):
        super(Customized_Executor, self).__init__(args)

    def get_client_trainer(self, conf):
        return Customized_Client(conf)

    def init_model(self):
        """Return the model architecture used in training"""
        model = tf.keras.applications.resnet.ResNet50(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=[32, 32, 3],
            pooling=None,
            classes=10
        )
        return model


if __name__ == "__main__":
    executor = Customized_Executor(parser.args)
    executor.run()

