#Copyright (C) 2021 Fanwei Kong, Shawn C. Shadden, University of California, Berkeley

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import models

from math import pi
from math import cos
from math import floor
class CosineAnnealingLearningRateSchedule(callbacks.Callback):
    # constructor
    def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.lrates = list()
 
    # calculate learning rate for an epoch
    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
        epochs_per_cycle = floor(n_epochs/n_cycles)
        cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return lrate_max/2 * (cos(cos_inner) + 1)
 
    # calculate and set learning rate at the start of the epoch
    def on_epoch_begin(self, epoch, logs=None):
        # calculate learning rate
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
        # set learning rate
        K.set_value(self.model.optimizer.lr, lr)
        # log value
        self.lrates.append(lr)

class SaveModelOnCD(callbacks.Callback):
    def __init__(self, keys, model_save_path, patience, grid_weight=None, grid_key=None):
        self.keys = keys
        self.save_path = model_save_path
        self.no_improve = 0
        self.patience = patience
        self.grid_weight = grid_weight
        self.grid_key = grid_key
    
    def on_train_begin(self, logs=None):
        self.best = np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        CD_val_loss = 0.
        for key in self.keys:
            CD_val_loss += logs.get('val_'+key+'_point_loss_cf')
        if self.grid_weight is not None:
            mean_CD_loss = CD_val_loss/float(len(self.keys))
            grid_loss = logs.get('val_'+self.grid_key+'_loss')
            new_weight = min(mean_CD_loss / grid_loss * K.get_value(self.grid_weight), 1000)
            K.set_value(self.grid_weight, new_weight)
            print("Setting grid loss {} weight to: {}.".format('val_'+self.grid_key+'_loss', K.get_value(self.grid_weight) ))
        if CD_val_loss < self.best:
            print("\nValidation loss decreased from %f to %f, saving model to %s.\n" % (self.best, CD_val_loss, self.save_path))
            self.best = CD_val_loss
            if isinstance(self.model.layers[-2], models.Model):
                self.model.layers[-2].save_weights(self.save_path)
            else:
                self.model.save_weights(self.save_path)
            self.no_improve = 0
        else:
            print("\nValidation loss did not improve from %f.\n" % self.best)
            self.no_improve += 1
        if self.no_improve > self.patience:
            self.model.stop_training = True

class ReduceLossWeight(callbacks.Callback):
    def __init__(self, grid_weight, patience=10, factor=0.5):
        self.grid_weight = grid_weight
        self.num_epochs = 0
        self.patience = patience
        self.factor = factor
    
    def on_epoch_end(self, epoch, logs=None):
        self.num_epochs += 1
        if self.num_epochs >= self.patience:
            new_weight = max(K.get_value(self.grid_weight) * self.factor, 10.)
            K.set_value(self.grid_weight, new_weight)
            print("Setting grid loss weight to: {}.".format(K.get_value(self.grid_weight)))
            self.num_epochs = 0

class AlternateLoss(callbacks.Callback):
    def __init__(self, weights, alter_num, start_epoch):
        self.weights = weights
        self.alter_num = alter_num
        self.weights_copy = K.get_value(weights)
        self.start_epoch = start_epoch
    def on_epoch_begin(self, epoch, logs=None):
        if epoch > self.start_epoch:
            if (epoch // self.alter_num) % 2 == 0:
                #set point loss
                weights = [self.weights_copy[0], 0., 0., 0., self.weights_copy[-1]]
            else:
                weights = [0., self.weights_copy[1], self.weights_copy[2], self.weights_copy[3], self.weights_copy[-1]]
            K.set_value(self.weights, weights)
        print("Current weights: ", epoch, K.get_value(self.weights))
