# Base class that all dataset models will inherit from

from __future__ import print_function
import nd_aggregation
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import random
import argparse
import byzantine
import os
import json
import gluonnlp

from abc import ABC, abstractmethod

class Module(ABC):
    @abstractmethod
    def createModel(self):
        return None

    @abstractmethod
    def loadTrainingData(self, ctx):
        # Return training data and masks (currently masks only for REDDIT dataset)
        return None, None
    
    @abstractmethod
    def loadTestingData(self):
        return None

    def getAccuracyMetric(self):
        # Default accuracy if not otherwise specified
        return mx.metric.Accuracy()

    def getPredictionsFromNetworkOutput(self, output):
        # Default argmax on axis 1
        return nd.argmax(output, axis=1)

    def initializeModel(self, net, ctx):
        # Default model initialization with Xavier
        net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)

    def getLossFunction(self):
        # Default cross entropy loss function
        return gluon.loss.SoftmaxCrossEntropyLoss()