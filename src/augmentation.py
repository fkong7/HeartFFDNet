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

import tensorflow as tf

def shift_img(output_imVg, label_img, width_shift_range, height_shift_range):
  return output_img, label_img

"""## Flipping the image randomly"""
def flip_img(horizontal_flip, tr_img, label_img):
  return tr_img, label_img

"""## Rotate the image with random angle""" 
def rotate_img(rotation, tr_img, label_img):
    return tr_img, label_img

"""##Scale/shift the image intensity randomly"""

def changeIntensity_img(tr_img, label_img, change):
  if change:
    scale = tf.random_uniform([], change['scale'][0], change['scale'][1])
    shift = tf.random_uniform([], change['shift'][0], change['shift'][1])
    tr_img = tr_img*scale+shift
    tr_img = tf.clip_by_value(tr_img, -1., 1.)
  return tr_img, label_img

def changeIntensity_img2(tr_img,  change):
  if change:
    scale = tf.random_uniform([], change['scale'][0], change['scale'][1])
    shift = tf.random_uniform([], change['shift'][0], change['shift'][1])
    tr_img = tr_img*scale+shift
    tr_img = tf.clip_by_value(tr_img, -1., 1.)
  return tr_img

"""## Assembling our transformations into our augment function"""
def _augment(inputs, outputs,
             changeIntensity=False):
  img, transform , spacing = inputs
  img = changeIntensity_img2(img,changeIntensity)
  inputs = (img, transform, spacing)
  return inputs, outputs
