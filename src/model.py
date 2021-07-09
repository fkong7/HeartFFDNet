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
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers
from custom_layers import *
import numpy as np
import sys
#from loss import *

# TO-DO check the axis of BatchNorm, channel first or last?
class HeartDeepFFD(object):
    def __init__(self, batch_size, input_size, hidden_dim, feed_dict,amplify_factor=1., num_mesh=1, num_seg=1, num_block=3, if_warp_im=False, if_output_grid=False):
        super(HeartDeepFFD, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.batch_size = batch_size
        self.feed_dict = feed_dict
        self.amplify_factor = amplify_factor
        self.num_mesh = num_mesh
        self.num_seg = num_seg
        self.num_block = num_block
        self.if_warp_im = if_warp_im
        self.if_output_grid = if_output_grid
    def build_conv_ffd(self):
        image_inputs = layers.Input(self.input_size, batch_size=self.batch_size)
        features =  self._unet_isensee_encoder(image_inputs) 
        
        grid_coords = layers.Lambda(lambda x: self.feed_dict['grid_coords'])(image_inputs)
        adjs = [j for j in self.feed_dict['adjs']] 
        ffd_matrix_mesh = self.feed_dict['ffd_matrix_mesh']

        grid_coords_p = ExpandDim(axis=0)(grid_coords) 
        grid_coords_p = Tile((self.batch_size, 1, 1))(grid_coords_p)
        
        outputs = self._conv_decoder(features, grid_coords_p, ffd_matrix_mesh)
        return models.Model([image_inputs],outputs)
    def build_3dunet(self):
        image_inputs = layers.Input(self.input_size, batch_size=self.batch_size)
        features =  self._unet_isensee_encoder(image_inputs)
        output = self._unet_isensee_decoder(features, num_filters=[128, 64, 32, 16])
        return models.Model([image_inputs], outputs)

    def build_keras(self, mode='ffd'):
        image_inputs = layers.Input(self.input_size, batch_size=self.batch_size)
        features =  self._unet_isensee_encoder(image_inputs) 
        if self.num_seg >0:
            decoder =  self._unet_isensee_decoder(features)
        transform = layers.Input((4, 4), batch_size=self.batch_size)
        spacing = layers.Input((3,), batch_size=self.batch_size)
        
        #mesh_coords = layers.Input(tensor=self.feed_dict['tmplt_coords'], batch_size=self.batch_size)
        #grid_coords = layers.Input(tensor=self.feed_dict['grid_coords'], batch_size=self.batch_size)
        #rbf_matrix = layers.Input(tensor=self.feed_dict['rbf_matrix'])
        #adjs= [layers.Input(tensor=j, sparse=True, batch_size=self.batch_size) for j in self.feed_dict['adjs']]
        #ffd_matrix_mesh = layers.Input(tensor=self.feed_dict['ffd_matrix_mesh'], sparse=True, batch_size=self.batch_size)
        #ffd_matrix_grid = layers.Input(tensor=self.feed_dict['ffd_matrix_grid'], sparse=True, batch_size=self.batch_size)
        #mesh_coords = self.feed_dict['tmplt_coords']
        grid_coords = layers.Lambda(lambda x: self.feed_dict['grid_coords'])(image_inputs)
        sample_coords = layers.Lambda(lambda x: self.feed_dict['sample_coords'])(image_inputs)
        adjs = self.feed_dict['adjs']
        ffd_matrix_mesh = self.feed_dict['ffd_matrix_mesh']
        ffd_matrix_image = self.feed_dict['ffd_matrix_image']


        #print("Grid coords: ", grid_coords.get_shape().as_list())
        grid_coords_p = ExpandDim(axis=0)(grid_coords) 
        grid_coords_p = Tile((self.batch_size, 1, 1))(grid_coords_p)
        sample_coords_p = ExpandDim(axis=0)(sample_coords) 
        sample_coords_p = Tile((self.batch_size, 1, 1))(sample_coords_p)
        outputs = self._graph_decoder_bspline((image_inputs, features, grid_coords_p, sample_coords_p, ffd_matrix_mesh, ffd_matrix_image), self.hidden_dim, adjs, transform)
        if self.num_seg >0:
            outputs = [decoder]+ list(outputs)
        #return models.Model([image_inputs, transform, spacing, mesh_coords, grid_coords, ffd_matrix_mesh, ffd_matrix_grid]+[item for item in adjs],outputs)
        return models.Model([image_inputs, transform, spacing],outputs)
    
    def load_pre_trained_weights(self, new_model, old_model_fn, trainable=False):
        pre_trained_im = UNet3DIsensee(self.input_size, num_class=self.num_seg)
        pre_trained_im = pre_trained_im.build()
        for i, layer in enumerate(pre_trained_im.layers[:55]):
            if i==53:
                i = 54
            elif i==54:
                i = 56
            print(i, layer.name, new_model.layers[i].name)
            weights = layer.get_weights()
            new_model.layers[i].set_weights(weights)
            new_model.layers[i].trainable = trainable
        del pre_trained_im
        return new_model

    def _unet_isensee_encoder(self, inputs, num_filters=[16, 32, 64, 128, 256]):
        unet = UNet3DIsensee(self.input_size, num_class=self.num_seg)
        output0 = unet._context_module(num_filters[0], inputs, strides=(1,1,1))
        output1 = unet._context_module(num_filters[1], output0, strides=(2,2,2))
        output2 = unet._context_module(num_filters[2], output1, strides=(2,2,2))
        output3 = unet._context_module(num_filters[3], output2, strides=(2,2,2))
        output4 = unet._context_module(num_filters[4], output3, strides=(2,2,2))
        return (output0, output1, output2, output3, output4)
    def _unet_isensee_decoder(self, inputs, num_filters=[64, 32, 16, 4]):
        unet = UNet3DIsensee(self.input_size, num_class=self.num_seg)
        output0, output1, output2, output3, output4 = inputs
        decoder0 = unet._decoder_block(num_filters[0], [output3, output4])
        decoder1 = unet._decoder_block(num_filters[1], [output2, decoder0])
        decoder2 = unet._decoder_block(num_filters[2], [output1, decoder1])
        decoder3 = unet._decoder_block_last_simple(num_filters[3], [output0, decoder2])
        output0 = layers.Conv3D(unet.num_class, (1, 1, 1))(decoder3)
        output1 = layers.Conv3D(unet.num_class, (1, 1, 1))(decoder2)
        output2_up = layers.UpSampling3D(size=(2,2,2))(layers.Conv3D(unet.num_class, (1, 1, 1))(decoder1))

        output_sum = layers.Add()([output2_up, output1])
        output_sum = layers.UpSampling3D(size=(2,2,2))(output_sum)
        output = layers.Add()([output_sum, output0])
        #output_sum = layers.Add()([output_sum, output0])
        #output = layers.Softmax()(output_sum)
        return output
    def _graph_res_block(self, inputs, adjs, in_dim, hidden_dim):
        output = GraphConv(in_dim ,hidden_dim, act=tf.nn.relu, adjs=adjs)(inputs)
        output2 = GraphConv(in_dim, hidden_dim, act=tf.nn.relu, adjs=adjs)(output)
        return layers.Average()([inputs, output2])

    def _graph_conv_block(self, inputs, adjs, feature_dim, hidden_dim, coord_dim, num_blocks):
        output = GraphConv(feature_dim, hidden_dim, act=tf.nn.relu, adjs=adjs)(inputs)
        output_cat = self._graph_res_block(output, adjs, hidden_dim, hidden_dim)
        for _ in range(num_blocks):
            output_cat = self._graph_res_block(output_cat, adjs, hidden_dim, hidden_dim)
        output = GraphConv(hidden_dim, coord_dim, act=lambda x: x, adjs=adjs)(output_cat)
        #output = GraphConv(hidden_dim, coord_dim, act=tf.nn.tanh)([output_cat]+[i for i in adjs])
        return output, output_cat
    def _conv_decoder(self, inputs, grid_coords, ffd_matrix_mesh):
        unet = UNet3DIsensee(self.input_size, num_class=self.num_seg)
        output0, output1, output2, output3, output4 = inputs # output4: 8, output3: 16, output2: 32, output1: 64, output0:128
        decoder0 = unet._decoder_block(128, [output3, output4])
        decoder1 = unet._decoder_block(64, [output2, decoder0])
        decoder2 = unet._decoder_block(32, [output1, decoder1])
        if self.num_seg > 0:
            decoder3 = unet._decoder_block_last_simple(16, [output0, decoder2])
            output0 = layers.Conv3D(unet.num_class, (1, 1, 1))(decoder3)
            output1 = layers.Conv3D(unet.num_class, (1, 1, 1))(decoder2)
            output2_up = layers.UpSampling3D(size=(2,2,2))(layers.Conv3D(unet.num_class, (1, 1, 1))(decoder1))

        output_sum = layers.Add()([output2_up, output1])
        output_sum = layers.UpSampling3D(size=(2,2,2))(output_sum)
        output = layers.Add()([output_sum, output0])

        d_grid1 = layers.Conv3D(3, (1, 1, 1))(decoder0)
        d_grid1_rs = layers.Reshape((16**3, 3))(d_grid1)
        #d_grid1_rs = layers.Lambda(lambda x: tf.Print(x, [x, tf.reduce_mean(x, axis=1)], message='Grid1: '))(d_grid1_rs)
        grid1 = layers.Add()([grid_coords, ScalarMul(self.amplify_factor)(d_grid1_rs)])
        mesh1 =  FFD(ffd_matrix_mesh)(grid1)
        
        d_grid2 = layers.Conv3D(3, (1, 1, 1))(decoder1)
        d_grid2 = layers.AveragePooling3D(pool_size=(2, 2, 2))(d_grid2)
        d_grid2_rs = layers.Reshape((16**3, 3))(d_grid2)
        #d_grid2_rs = layers.Lambda(lambda x: tf.Print(x, [x, tf.reduce_mean(x, axis=1)], message='Grid2: '))(d_grid2_rs)
        grid2 = layers.Add()([grid1, ScalarMul(self.amplify_factor)(d_grid2_rs)])
        mesh2 =  FFD(ffd_matrix_mesh)(grid2)
        
        d_grid3 = layers.Conv3D(3, (1, 1, 1))(decoder2)
        d_grid3 = layers.AveragePooling3D(pool_size=(4, 4, 4))(d_grid3)
        d_grid3_rs = layers.Reshape((16**3, 3))(d_grid3)
        #d_grid3_rs = layers.Lambda(lambda x: tf.Print(x, [x, tf.reduce_mean(x, axis=1)], message='Grid3: '))(d_grid3_rs)
        grid3 = layers.Add()([grid2, ScalarMul(self.amplify_factor)(d_grid3_rs)])
        mesh3 =  FFD(ffd_matrix_mesh)(grid3)
        
        mesh1 = ScalarMul(128)(mesh1)
        mesh2 = ScalarMul(128)(mesh2)
        mesh3 = ScalarMul(128)(mesh3)
        if self.num_mesh > 1:
            output1_list = []
            output2_list = []
            output3_list = []
            for i in range(len(self.feed_dict['struct_node_ids'])-1):
                mesh1_i = layers.Lambda(lambda x: x[:, self.feed_dict['struct_node_ids'][i]:self.feed_dict['struct_node_ids'][i+1], :])(mesh1)
                mesh2_i = layers.Lambda(lambda x: x[:, self.feed_dict['struct_node_ids'][i]:self.feed_dict['struct_node_ids'][i+1], :])(mesh2)
                mesh3_i = layers.Lambda(lambda x: x[:, self.feed_dict['struct_node_ids'][i]:self.feed_dict['struct_node_ids'][i+1], :])(mesh3)
                output1_list.append(mesh1_i)
                output2_list.append(mesh2_i)
                output3_list.append(mesh3_i)
        output_all = [d_grid1_rs, d_grid2_rs, d_grid3_rs]+output1_list+output2_list+output3_list
        if self.num_seg>0:
            output_all = [output]+output_all
        return output_all
    def _graph_decoder_bspline(self, inputs,  hidden_dim, adjs, transform):
        coord_dim = 3
        if self.num_block==3:
            graph_conv_num = [384, 96, 48]
            graph_block_num = [256, 64, 32]
            feat_level = [[3,4], [1,2], [0,1]]
        elif self.num_block==1:
            graph_conv_num = [32]
            graph_block_num = [128]
            feat_level = [[4,3,2,1,0]]
        else:
            raise NotImplementedError
        image_inputs, features, grid_coords, sample_coords, ffd_matrix_mesh,  ffd_matrix_image = inputs
        input_size = [float(i) for  i in list(self.input_size)]
        curr_grid = grid_coords
        curr_feat = grid_coords
        curr_sample_space = sample_coords

        out_list = []
        out_d_grid_list = []
        out_image = []
        out_grid_list = []
        for l, (conv_num, block_num, feat) in enumerate(zip(graph_conv_num, graph_block_num, feat_level)):
            output =  GraphConv(curr_feat.get_shape().as_list()[-1], conv_num, act=tf.nn.relu, adjs=adjs[l])(curr_feat)
            output_feat = Projection(feat, input_size)([i for i in features]+[curr_sample_space])
            output = layers.Concatenate(axis=-1)([MatMul(self.feed_dict['grid_downsample'][l], sparse=True)(output_feat), output])
            output1_dx, curr_feat = self._graph_conv_block(output, adjs[l], output.get_shape().as_list()[-1], block_num, coord_dim, 3)
            
            output1_dx_scaled = ScalarMul(self.amplify_factor)(output1_dx)
            out_d_grid_list.append(output1_dx)
            if 0 < l : # if middle blocks, add additional deformation to mesh using down-sampled grid
                mesh1 = layers.Add()([mesh1, FFD(ffd_matrix_mesh[l])(output1_dx_scaled)])
            else:
                curr_grid = layers.Add()([curr_grid, output1_dx_scaled])
                mesh1 = FFD(ffd_matrix_mesh[l])(curr_grid)
            curr_sample_space = mesh1
            # upsample and prepare for next block
            if l < self.num_block - 1:
                curr_feat = MatMul(self.feed_dict['grid_upsample'][l], sparse=True)(curr_feat) 
            mesh1_scaled = ScalarMul(128)(mesh1)
            if self.num_mesh > 1:
                output1_list = []
                for i in range(len(self.feed_dict['struct_node_ids'])-1):
                    mesh1_i = layers.Lambda(lambda x: x[:, self.feed_dict['struct_node_ids'][i]:self.feed_dict['struct_node_ids'][i+1], :])(mesh1_scaled)
                    output1_list.append(mesh1_i)
                out_list += output1_list
            else:
                mesh_i = layers.Lambda(lambda x: x)(mesh1_scaled)
                out_list.append(mesh_i)
        return out_image + out_d_grid_list + out_list

class UNet3DIsensee(object):
    def __init__(self, input_size, num_class=8, num_filters=[16, 32, 64, 128, 256]):
        super(UNet3DIsensee, self).__init__()
        self.num_class = num_class
        self.input_size = input_size
        self.num_filters = num_filters
    
    def build(self):
        inputs = layers.Input(self.input_size)

        output0 = self._context_module(self.num_filters[0], inputs, strides=(1,1,1))
        output1 = self._context_module(self.num_filters[1], output0, strides=(2,2,2))
        output2 = self._context_module(self.num_filters[2], output1, strides=(2,2,2))
        output3 = self._context_module(self.num_filters[3], output2, strides=(2,2,2))
        output4 = self._context_module(self.num_filters[4], output3, strides=(2,2,2))
        
        decoder0 = self._decoder_block(self.num_filters[3], [output3, output4])
        decoder1 = self._decoder_block(self.num_filters[2], [output2, decoder0])
        decoder2 = self._decoder_block(self.num_filters[1], [output1, decoder1])
        decoder3 = self._decoder_block_last(self.num_filters[0], [output0, decoder2])
        output0 = layers.Conv3D(self.num_class, (1, 1, 1))(decoder3)
        output1 = layers.Conv3D(self.num_class, (1, 1, 1))(decoder2)
        output2_up = layers.UpSampling3D(size=(2,2,2))(layers.Conv3D(self.num_class, (1, 1, 1))(decoder1))

        output_sum = layers.Add()([output2_up, output1])
        output_sum = layers.UpSampling3D(size=(2,2,2))(output_sum)
        output_sum = layers.Add()([output_sum, output0])
        output = layers.Softmax()(output_sum)

        return models.Model(inputs=[inputs], outputs=[output])

    def _conv_block(self, num_filters, inputs, strides=(1,1,1)):
        output = layers.Conv3D(num_filters, (3, 3, 3),kernel_regularizer=regularizers.l2(0.01),  padding='same', strides=strides)(inputs)
        output = layers.LeakyReLU(alpha=0.01)(InstanceNormalization(axis=-1)(output))
        return output

    def _context_module(self, num_filters, inputs, dropout_rate=0.3, strides=(1,1,1)):
        conv_0 = self._conv_block(num_filters, inputs, strides=strides)
        conv_1 = self._conv_block(num_filters, conv_0)
        dropout = layers.SpatialDropout3D(rate=dropout_rate)(conv_1)
        conv_2 = self._conv_block(num_filters, dropout)
        sum_output = layers.Add()([conv_0, conv_2])
        return sum_output
    
    def _decoder_block(self, num_filters,  inputs, strides=(2,2,2)):
        features, encoder_out = inputs
        upsample = layers.UpSampling3D(size=strides)(encoder_out)
        conv_1 = self._conv_block(num_filters, upsample)
        concat = layers.Concatenate(axis=-1)([conv_1, features])
        conv_2 = self._conv_block(num_filters, concat)
        conv_3 = layers.Conv3D(num_filters, (1,1,1), padding='same')(conv_2)
        output = layers.LeakyReLU(alpha=0.01)(InstanceNormalization(axis=-1)(conv_3))
        return output
    
    def _decoder_block_last_simple(self, num_filters,  inputs, strides=(2,2,2)):
        features, encoder_out = inputs
        upsample = layers.UpSampling3D(size=strides)(encoder_out)
        conv_1 = self._conv_block(num_filters, upsample)
        concat = layers.Concatenate(axis=-1)([conv_1, features])
        conv_2 = self._conv_block(num_filters, concat)
        return conv_2

    def _decoder_block_last(self, num_filters,  inputs, strides=(2,2,2)):
        features, encoder_out = inputs
        upsample = layers.UpSampling3D(size=strides)(encoder_out)
        conv_1 = self._conv_block(num_filters, upsample)
        concat = layers.Concatenate(axis=-1)([conv_1, features])
        conv_2 = self._conv_block(num_filters*2, concat)
        return conv_2
    
