# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import torch
import h5py
import numpy as np
import imageio
from options.train_options import TrainOptions
from loaders import aligned_data_loader
from models import pix2pix_model

BATCH_SIZE = 1

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

eval_TUM_list_path = 'test_data/test_tum_hdf5_list.txt'

isTrain = False
eval_num_threads = 1
eval_data_loader = aligned_data_loader.TUMDataLoader(opt, eval_TUM_list_path,
                                                     isTrain, BATCH_SIZE,
                                                     eval_num_threads)
dataset = eval_data_loader.load_data()
data_size = len(eval_data_loader)
print('========================= TUM evaluation #images = %d =========' %
      data_size)

model = pix2pix_model.Pix2PixModel(opt, False)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch = 0
global_step = 0

count = 0.0

print(
    '============================= TUM INFERENCE ============================'
)
model.switch_to_eval()

for i, data in enumerate(dataset):
    print(i)
    stacked_img = data[0]
    targets = data[1]

    model.eval_save_tum_img(stacked_img, targets, 'test_data/viz_predictions/')
    count += stacked_img.size(0)



tum_result_dir = 'test_data/viz_predictions/tum_hdf5/'
if not os.path.exists(tum_result_dir + 'jpegs/'):
                os.makedirs(tum_result_dir + 'jpegs/')

h5files = [f for f in os.listdir(tum_result_dir) if os.path.isfile(tum_result_dir + f)]

#convert h5 results to jpg
for h5filename in h5files:
    f = h5py.File(tum_result_dir + h5filename, 'r')
    prediction = np.array(f['prediction/pred_depth'][:,:])
    imageio.imwrite(tum_result_dir + 'jpegs/' + h5filename[:-3], prediction)


