# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# verify
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import time
import sys
import copy
from   datetime import datetime
import collections
import json
import operator
import os

class Net_Trainer:
    
    def __init__(self):
        self.best_trace      = collections.OrderedDict()
        self.dataset         = collections.OrderedDict()
        self.training_trace  = collections.OrderedDict()
        self.best_arch       = None
        self.best_acc        = 0
        self.best_accuracy   = 0
        self.counter         = 0

        raw_data = []
        with open('/Users/linnan/Documents/LA-MCTS/datasets_comparisions/ConvNet/cnn_int_dataset', 'r') as infile:
            self.dataset = json.loads( infile.read() )
        for arch in self.dataset:
            acc = self.dataset[arch]
            if acc > self.best_acc:
                self.best_acc  = acc
                self.best_arch = json.dumps( arch )
        print("searching target:", self.best_arch," acc:", self.best_acc)
        print("trainer loaded:", len(self.dataset)," entries" )
    
    def print_best_traces(self):
        print("%"*20)
        print("=====> best accuracy so far:", self.best_accuracy)
        sorted_best_traces = sorted(self.best_trace.items(), key=operator.itemgetter(1))
        for item in sorted_best_traces:
            print(item[0],"==>", item[1])
        for item in sorted_best_traces:
            print(item[1])
        print("%"*20)
       
    def train_net(self, network):
        # input is a code of an architecture
        assert type( network ) == type( [] )
        network_str = json.dumps( network )
        assert type(self.dataset[network_str])== type(float(0.1))
        assert network_str in self.dataset
        is_found = False
        acc = self.dataset[network_str]
        # we ensure not to repetitatively sample same architectures
        assert network_str not in self.training_trace.keys()
        self.training_trace[network_str] = acc
        self.counter += 1
        result              = {}
        result[network_str] = acc 
        if acc > self.best_accuracy:
            print("@@@update best state:", network)
            print("@@@update best acc:", acc)
            print("target acc:", self.best_acc)
            self.best_accuracy = acc
            item = [acc, self.counter]
            self.best_trace[network_str] = item
            if acc == self.best_acc:
                sorted_best_traces = sorted(self.best_trace.items(), key=operator.itemgetter(1))
                final_results = []
                for item in sorted_best_traces:
                    final_results.append( item[1] )
                final_results_str = json.dumps(final_results)
                with open("result.txt", "a") as f:
                    f.write(final_results_str + '\n')
                print("$$$$$$$$$$$$$$$$$$$CONGRATUGLATIONS$$$$$$$$$$$$$$$$$$$")
                os._exit(1)

        return result
