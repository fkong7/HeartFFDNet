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
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers
import tf_utils
import numpy as np
def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
#        res = tf.sparse_tensor_dense_matmul(x, y)
        res = tf_utils.sparse_tensor_dense_tensordot(x, y, axes=[[1], [1]])
        res = tf.transpose(res, perm=[1,0,2])
    else:
#        res = tf.matmul(x, y)
        res = tf.tensordot(x, y, axes=1)
        #res = tf.transpose(res, perm=[1,0,2])
    return res

from math import factorial

def comb(n, k):
    return factorial(n) / factorial(k) / factorial(n - k)

class MatMul(layers.Layer):
    def __init__(self,matrix, sparse=False, **kwargs):
        super(MatMul, self).__init__(**kwargs)
        self.matrix = matrix
        self.sparse = sparse
    def get_config(self):
        config = {'matrix': self.matrix, 'sparse': self.sparse}
        base_config = super(MatMul, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def build(self, input_shape):
        super(MatMul, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        if self.sparse:
            output = dot(self.matrix, x, self.sparse)
        else:
            matrix = tf.expand_dims(tf.constant(self.matrix, tf.float32), axis=0)
            output = tf.matmul(matrix, x)
        return output
    def compute_output_shape(self, input_shape):
        output_shape = tf.identity(input_shape)
        output_shape[1] = self.matrix.get_shape().as_list()[1]
        return output_shape 

class FFD(layers.Layer):
    def __init__(self,ffd_matrix,scale_vec=None, offset=None, **kwargs):
        super(FFD, self).__init__(**kwargs)
        self.ffd_matrix = ffd_matrix
        self.scale_vec = tf.expand_dims(tf.constant(scale_vec, tf.float32), axis=0) if scale_vec is not None else None
        self.offset = tf.expand_dims(tf.constant(offset, tf.float32), axis=0) if offset is not None else None
    def get_config(self):
        config = {'ffd_matrix': self.ffd_matrix, 'scale_vec': self.scale_vec,
                'offset': self.offset}
        base_config = super(FFD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def build(self, input_shape):
        self.batch_size = input_shape[0]
        super(FFD, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        if self.scale_vec is not None:
            x = x * self.scale_vec
        if self.offset is not None:
            x = x - self.offset
        dx = dot(self.ffd_matrix, x, sparse=True)
        return dx
    def compute_output_shape(self, input_shape):
        output_shape = tf.identity(input_shape)
        output_shape[1] = self.ffd_matrix.get_shape().as_list()[0]
        return output_shape 

class RBFD(layers.Layer):
    def __init__(self,rbf_matrix, **kwargs):
        self.rbf_matrix = rbf_matrix
        super(RBFD, self).__init__(**kwargs)
    def get_config(self):
        config = {'rbf_matrix': self.rbf_matrix}
        base_config = super(RBFD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.batch_size = input_shape[0][0]
        super(RBFD, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        d_grid_coords, mesh_coords= x
        d_mesh = tf.matmul(tf.expand_dims(tf.constant(self.rbf_matrix, dtype=tf.float32), axis=0), d_grid_coords)
        deformed = mesh_coords + d_mesh
        return deformed    
    def compute_output_shape(self, input_shape):
        return input_shape[-1]

class Tile(layers.Layer):
    def __init__(self, repeats,**kwargs):
        super(Tile, self).__init__(**kwargs)
        self.repeats = repeats
    def get_config(self):
        config = {'repeats': self.repeats}
        base_config = super(Tile, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def call(self, x):
        x = tf.tile(x, self.repeats)
        return x

class Gather(layers.Layer):
    def __init__(self, gather_index, **kwargs):
        super(Gather, self).__init__(**kwargs)
        self.gather_index = gather_index
    def get_config(self):
        config = {'gather_index': self.gather_index}
        base_config = super(Gather, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def build(self, input_shape):
        super(Gather, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        sample = tf.gather(x, tf.constant(self.gather_index, dtype=tf.int32), axis=1)
        return sample

class Translate(layers.Layer):
    def __init__(self, **kwargs):
        super(Translate, self).__init__(**kwargs)
    def get_config(self):
        base_config = super(Translate, self).get_config()
        return dict(list(base_config.items()))
    def call(self, x):
        trans, coords = x
        batch_size = coords.get_shape().as_list()[0]
        trans = tf.reshape(trans, [batch_size, 1, 3])
        coords += trans
        center  = tf.reduce_mean(coords, axis=1, keepdims=True)
        #center = tf.Print(center, [trans, center], message="trans, center")
        return [center, coords]
    def build(self, input_shape):
        super(Translate, self).build(input_shape)  # Be sure to call this at the end
    def compute_output_shape(self, input_shape):
        return input_shape

class ExpandDim(layers.Layer):
    def __init__(self, axis=0,**kwargs):
        super(ExpandDim, self).__init__(**kwargs)
        self.axis = axis
    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(ExpandDim, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def call(self, x):
        x = tf.expand_dims(x, axis=self.axis)
        return x
    def compute_output_shape(self, input_shape):
        if self.axis==-1:
            input_shape = list(input_shape).append(1)
        else:
            input_shape = list(input_shape).insert(self.axis, 1)
        return input_shape

class Split(layers.Layer):
    def __init__(self, axis=-1, num=1,**kwargs):
        super(Split, self).__init__(**kwargs)
        self.axis = axis
        self.num = num
    def get_config(self):
        config = {'axis': self.axis, 'num': self.num}
        base_config = super(Split, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def build(self, input_shape):
        super(Split, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        x = tf.split(x, self.num, axis=self.axis)
        return x
    def compute_output_shape(self, input_shape):
        shape_list = []
        input_shape[self.axis] = input_shape[self.axis] // self.num
        for i in range(self.num):
            shape_list.append(input_shape)
        return shape_list

class ImageWarped(layers.Layer):
    def __init__(self, size=[128,128,128], **kwargs):
        super(ImageWarped, self).__init__(**kwargs)
        self.size = size
    def get_config(self):
        config = {'size': self.size}
        base_config = super(ImageWarped, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # input shapes: 4 by 4 transform matrix, feature 1, 2, 3, 4, mesh_coords
        assert isinstance(input_shape, list)
        self.batch_size = input_shape[-1][0]
        super(ImageWarped, self).build(input_shape)  # Be sure to call this at the end
    
    def call(self, inputs):
        image_inputs, image_grid = inputs
        factor = tf.constant([[[self.size[0], self.size[1], self.size[2]]]], dtype=tf.float32)
        # TODO: change to zero diff when oob
        #indices = tf.Print(indices, [tf.reduce_min(indices, axis=1), tf.reduce_max(indices, axis=1)], message="Before: ")
        indices = image_grid * factor 
        indices = tf.clip_by_value(indices, clip_value_min=0.001, clip_value_max=tf.reduce_min(factor)-1.001)
        #indices = tf.Print(indices, [tf.reduce_min(indices, axis=1), tf.reduce_max(indices, axis=1)], message="After: ")
        x1 = tf.floor(indices[:,:,0])
        x2 = tf.ceil(indices[:,:,0])
        y1 = tf.floor(indices[:,:,1])
        y2 = tf.ceil(indices[:,:,1])
        z1 = tf.floor(indices[:,:,2])
        z2 = tf.ceil(indices[:,:,2])
        q11 = gather_nd(image_inputs, tf.cast(tf.stack([x1, y1, z1], axis=-1), tf.int32))
        #q11 = tf.Print(q11, [indices, q11])
        q21 = gather_nd(image_inputs, tf.cast(tf.stack([x2, y1, z1], axis=-1), tf.int32))
        q12 = gather_nd(image_inputs, tf.cast(tf.stack([x1, y2, z1], axis=-1), tf.int32))
        q22 = gather_nd(image_inputs, tf.cast(tf.stack([x2, y2, z1], axis=-1), tf.int32))
        wx = tf.expand_dims(tf.subtract(indices[:,:,0], x1), -1)
        wx2 = tf.expand_dims(tf.subtract(x2, indices[:,:,0]), -1)
        lerp_x1 = tf.add(tf.multiply(q21, wx), tf.multiply(q11, wx2))
        lerp_x2 = tf.add(tf.multiply(q22, wx), tf.multiply(q12, wx2))
        wy = tf.expand_dims(tf.subtract(indices[:,:,1], y1), -1)
        wy2 = tf.expand_dims(tf.subtract(y2, indices[:,:,1]), -1)
        lerp_y1 = tf.add(tf.multiply(lerp_x2, wy), tf.multiply(lerp_x1, wy2))

        q11 = gather_nd(image_inputs, tf.cast(tf.stack([x1, y1, z2], axis=-1), tf.int32))
        q21 = gather_nd(image_inputs, tf.cast(tf.stack([x2, y1, z2], axis=-1), tf.int32))
        q12 = gather_nd(image_inputs, tf.cast(tf.stack([x1, y2, z2], axis=-1), tf.int32))
        q22 = gather_nd(image_inputs, tf.cast(tf.stack([x2, y2, z2], axis=-1), tf.int32))
        lerp_x1 = tf.add(tf.multiply(q21, wx), tf.multiply(q11, wx2))
        lerp_x2 = tf.add(tf.multiply(q22, wx), tf.multiply(q12, wx2))
        lerp_y2 = tf.add(tf.multiply(lerp_x2, wy), tf.multiply(lerp_x1, wy2))

        wz = tf.expand_dims(tf.subtract(indices[:,:,2], z1), -1)
        wz2 = tf.expand_dims(tf.subtract(z2, indices[:,:,2]),-1)
        deformed = tf.add(tf.multiply(lerp_y2, wz), tf.multiply(lerp_y1, wz2))
        return deformed
    def compute_output_shape(self, input_shape):
        output_shape = tf.identity(input_shape[-1])
        output_shape[-1] = 1
        return output_shape

class SplitMeshByIDs(layers.Layer):
    def __init__(self,id_list,**kwargs):
        super(SplitMeshByIDs, self).__init__(**kwargs)
        self.id_list = id_list
    
    def get_config(self):
        config = {'id_list': self.id_list}
        base_config = super(SplitMeshByIDs, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shape):
        super(SplitMeshByIDs, self).build(input_shape)  # Be sure to call this at the end
    
    def call(self, x):
        out_list = []
        for i in range(len(self.id_list)-1):
            x_i = x[:, self.id_list[i]:self.id_list[i+1], :]
            out_list.append(x_i)
        return out_list

    def compute_output_shape(self, input_shape):
        shape_list = []
        for i in range(len(self.id_list)-1):
            mesh_shape = tf.identity(input_shape)
            mesh_shape[1] = self.id_list[i+1]-self.id_list[i]
            shape_list.append(mesh_shape)
        return shape_list

class ScalarMul(layers.Layer):
    def __init__(self, factor=1.,**kwargs):
        super(ScalarMul, self).__init__(**kwargs)
        self.factor = factor
    def get_config(self):
        config = {'factor': self.factor}
        base_config = super(ScalarMul, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def call(self, x):
        x *= self.factor
        return x
def gather_nd(features, indices):
    # tf1.12 does not support gather_nd with batch_dims; work around: 
    ind_shape = tf.shape(indices)
    indices = tf.reshape(indices, [ind_shape[0]*ind_shape[1], ind_shape[2]])
    first = tf.cast(tf.range(tf.size(indices[:,0]))/ind_shape[1], dtype=tf.int32)
    indices = tf.concat([tf.expand_dims(first, axis=-1), indices], axis=1)
    gather = tf.reshape(tf.gather_nd(features, indices), [ind_shape[0],ind_shape[1],tf.shape(features)[-1]])
    return gather

class Projection(layers.Layer):
    def __init__(self, feature_block_ids=[1], size=128, **kwargs):
        super(Projection, self).__init__(**kwargs)
        self.feature_block_ids = feature_block_ids
        self.size = size
    def get_config(self):
        config = {'feature_block_ids': self.feature_block_ids, 'size': self.size}
        base_config = super(Projection, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # input shapes: 4 by 4 transform matrix, feature 1, 2, 3, 4, mesh_coords
        assert isinstance(input_shape, list)
        self.batch_size = input_shape[-1][0]
        super(Projection, self).build(input_shape)  # Be sure to call this at the end
    
    def call(self, inputs):
        features0,features1,features2,features3,features4, mesh_coords= inputs
        mesh_shape = mesh_coords.get_shape().as_list()
        mesh_coords = tf.reshape(mesh_coords, [mesh_shape[0], mesh_shape[1]*(mesh_shape[2]//3), 3])
        out = tf.zeros([mesh_shape[0], mesh_shape[1]*(mesh_shape[2]//3), 0], tf.float32)
        features = [features0, features1, features2, features3, features4]
        num = len(features)
        id_list = self.feature_block_ids
        features = [features[i] for i in self.feature_block_ids]
        for i, power in enumerate(id_list):
            factor = tf.constant([[[(0.5**power)*self.size[0], (0.5**power)*self.size[1], (0.5**power)*self.size[2]]]], dtype=tf.float32)
            factor = tf.tile(factor, [tf.shape(mesh_coords)[0], 1,1])
            indices = mesh_coords * factor
            #indices = tf.cast(indices[:,:,:-1], tf.int32)
            #indices = tf.Print(indices, [self.feature_block_ids, indices, mesh_coords], message='Level: %d' % i)
            indices = tf.clip_by_value(indices, 0.01,tf.cast(tf.reduce_min(tf.shape(features[i])[1:4]), tf.float32)-1.01)
            x1 = tf.floor(indices[:,:,0])
            x2 = tf.ceil(indices[:,:,0])
            y1 = tf.floor(indices[:,:,1])
            y2 = tf.ceil(indices[:,:,1])
            z1 = tf.floor(indices[:,:,2])
            z2 = tf.ceil(indices[:,:,2])
            q11 = gather_nd(features[i], tf.cast(tf.stack([x1, y1, z1], axis=-1), tf.int32))
            #q11 = tf.Print(q11, [indices, q11])
            q21 = gather_nd(features[i], tf.cast(tf.stack([x2, y1, z1], axis=-1), tf.int32))
            q12 = gather_nd(features[i], tf.cast(tf.stack([x1, y2, z1], axis=-1), tf.int32))
            q22 = gather_nd(features[i], tf.cast(tf.stack([x2, y2, z1], axis=-1), tf.int32))
            wx = tf.expand_dims(tf.subtract(indices[:,:,0], x1), -1)
            wx2 = tf.expand_dims(tf.subtract(x2, indices[:,:,0]), -1)
            lerp_x1 = tf.add(tf.multiply(q21, wx), tf.multiply(q11, wx2))
            lerp_x2 = tf.add(tf.multiply(q22, wx), tf.multiply(q12, wx2))
            wy = tf.expand_dims(tf.subtract(indices[:,:,1], y1), -1)
            wy2 = tf.expand_dims(tf.subtract(y2, indices[:,:,1]), -1)
            lerp_y1 = tf.add(tf.multiply(lerp_x2, wy), tf.multiply(lerp_x1, wy2))

            q11 = gather_nd(features[i], tf.cast(tf.stack([x1, y1, z2], axis=-1), tf.int32))
            q21 = gather_nd(features[i], tf.cast(tf.stack([x2, y1, z2], axis=-1), tf.int32))
            q12 = gather_nd(features[i], tf.cast(tf.stack([x1, y2, z2], axis=-1), tf.int32))
            q22 = gather_nd(features[i], tf.cast(tf.stack([x2, y2, z2], axis=-1), tf.int32))
            lerp_x1 = tf.add(tf.multiply(q21, wx), tf.multiply(q11, wx2))
            lerp_x2 = tf.add(tf.multiply(q22, wx), tf.multiply(q12, wx2))
            lerp_y2 = tf.add(tf.multiply(lerp_x2, wy), tf.multiply(lerp_x1, wy2))

            wz = tf.expand_dims(tf.subtract(indices[:,:,2], z1), -1)
            wz2 = tf.expand_dims(tf.subtract(z2, indices[:,:,2]),-1)
            lerp_z = tf.add(tf.multiply(lerp_y2, wz), tf.multiply(lerp_y1, wz2))
            out = tf.concat([out, lerp_z], axis=-1)
        #out -= tf.reduce_mean(out, [1,2], keepdims=True)
        out = tf.reshape(out, [mesh_shape[0], mesh_shape[1], out.get_shape().as_list()[-1]*(mesh_shape[2]//3)])
        return out

class GraphConv(layers.Layer):
    def __init__(self, input_dim=10, output_dim=10, adjs=None, dropout=False,
                 sparse_inputs=False, act=tf.nn.relu, bias=True,
                 featureless=False, **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.act = act
        self.featureless = featureless
        self.vars = {}
        self.adjs = adjs
    def get_config(self):
        config = {'input_dim': self.input_dim, 
                'output_dim': self.output_dim, 
                'adjs': self.adjs,
                'dropout':self.dropout, 
                'sparse_inputs': self.sparse_inputs,
                'act': self.act, 
                'bias':self.bias, 
                'featureless': self.featureless}
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.batch_size = input_shape[0]
        num_cheb_support = len(self.adjs)
        for i in range(1, num_cheb_support+1):
            name = 'kernel_'+str(i)
            self.vars[name] = self.add_weight(name=name, 
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_normal',
                                      regularizer=regularizers.l2(0.01), 
                                      trainable=True)
        self.vars['bias'] = self.add_weight(name='bias', 
                                      shape=( self.output_dim),
                                      initializer='zeros',
                                      #regularizer=regularizers.l2(0.01), 
                                      trainable=True)
        super(GraphConv, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        support_1 = dot(x, self.vars['kernel_1'], sparse=self.sparse_inputs)
        output = dot(self.adjs[0], support_1, sparse=True)
        for i in range(2, len(self.adjs)+1):
            name = 'kernel_'+str(i)
            support = dot(x, self.vars[name], sparse=self.sparse_inputs)
            output = output + dot(self.adjs[i-1], support, sparse=True)
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

    def compute_output_shape(self, input_shape):
        output_shape = tf.identity(input_shape)
        output_shape[-1] = self.output_dim
        return output_shape

from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.python.keras import initializers, regularizers, constraints

class InstanceNormalization(layers.Layer):
    """Instance normalization layer. Taken from keras.contrib
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvBlock(layers.Layer):
    def __init__(self, num_filters):
        super(ConvBlock, self).__init__()
        self.conv_3d = layers.Conv3D(num_filters, (3, 3, 3), padding='same')
        self.conv_3d_2 = layers.Conv3D(num_filters, (3, 3, 3), padding='same')
        self.batch_norm = layers.BatchNormalization()
        self.batch_norm_2 = layers.BatchNormalization()
        self.activation = layers.Activation('relu')

    def call(self, inputs):
        encoder = self.conv_3d(inputs)
        encoder = self.batch_norm(encoder)
        encoder = self.activation(encoder)
        encoder = self.conv_3d_2(encoder)
        encoder = self.batch_norm_2(encoder)
        return self.activation(encoder)

class EncoderBlock(layers.Layer):
    def __init__(self, num_filters):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(num_filters)
        self.pool = layers.MaxPooling3D((2, 2, 2), strides=(2,2,2))

    def call(self, inputs):
        encoder = self.conv_block(inputs)
        return [self.pool(encoder), encoder]

class DecoderBlock(layers.Layer):
    def __init__(self, num_filters):
        super(DecoderBlock, self).__init__()
        self.convT = layers.Conv3DTranspose(num_filters, (2, 2, 2), strides=(2, 2, 2), padding='same')
        self.conv2 = layers.Conv3D(int(num_filters/2), (3, 3, 3), padding='same')
        self.conv1 = layers.Conv3D(int(num_filters/2), (3, 3, 3), padding='same')
        self.concate = layers.Concatenate(axis=-1)
        self.batch_norm = layers.BatchNormalization()
        self.batch_norm_2 = layers.BatchNormalization()
        self.batch_norm_3 = layers.BatchNormalization()
        self.activation = layers.Activation('relu')
    
    def call(self, inputs):
        features, encoder_out = inputs
        decoder_out = self.convT(encoder_out)
        decoder_out = self.concate([decoder_out, features])
        decoder_out = self.activation(self.batch_norm(decoder_out))
        decoder_out = self.activation(self.batch_norm_2(self.conv2(decoder_out)))
        decoder_out = self.activation(self.batch_norm_3(self.conv1(decoder_out)))
        return decoder_out


class UnetEncoder(layers.Layer):
    def __init__(self):
        super(UnetEncoder, self).__init__()
        self.block1 = EncoderBlock(32)
        self.block2 = EncoderBlock(64)
        self.block3 = EncoderBlock(128)
        self.center = ConvBlock(256)

    def call(self, inputs):
        encoder0_pool, encoder0= self.block1(inputs)
        encoder1_pool, encoder1 = self.block2(encoder0_pool)
        encoder2_pool, encoder2 = self.block3(encoder1_pool)
        center = self.center(encoder2_pool)
        return [encoder0, encoder1, encoder2, center]

class UNetDecoder(layers.Layer):
    def __init__(self, num_class):
        super(UNetDecoder, self).__init__()
        self.block1 = DecoderBlock(512)
        self.block2 = DecoderBlock(256)
        self.block3 = DecoderBlock(128)
        self.conv = layers.Conv3D(num_class, (1,1,1), activation='softmax', data_format="channels_last")

    def call(self, inputs):
        encoder0, encoder1, encoder2, center = inputs
        decoder0 = self.block1([encoder2, center])
        decoder1 = self.block2([encoder1, decoder0])
        decoder2 = self.block3([encoder0, decoder1])
        output = self.conv(decoder2)
        return output
