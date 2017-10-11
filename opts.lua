--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 ResNet Training script')
   cmd:text('See https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md for examples')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-data',       '',         'Path to dataset')
   cmd:option('-dataset',    'imagenet', 'Options: imagenet | cifar10 | cifar100')
   cmd:option('-manualSeed', 0,          'Manually set RNG seed')
   cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
   cmd:option('-gen',        'gen',      'Path to save generated files')
   cmd:option('-precision', 'single',    'Options: single | double | half')
   ------------- Data options ------------------------
   cmd:option('-nThreads',        2,        'number of data loading threads')
   cmd:option('-DataAug',         'true',   'use data augmentation or not')
   cmd:option('-validset',        'true',   'use validation set or not')
   ------------- Training options --------------------
   cmd:option('-nEpochs',         0,       'Number of total epochs to run')
   cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',       32,      'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',        'false', 'Run on validation set only')
   cmd:option('-tenCrop',         'false', 'Ten-crop testing')
   ------------- Checkpointing options ---------------
   cmd:option('-save',            'checkpoints', 'Directory in which to save checkpoints')
   cmd:option('-resume',          'none',        'Resume from the latest checkpoint in this directory')

   ---------- Optimization options ----------------------
   cmd:option('-LR',              0.1,   'initial learning rate')
   cmd:option('-momentum',        0.9,   'momentum')
   cmd:option('-weightDecay',     1e-4,  'weight decay')
   ---------- Model options ----------------------------------
   cmd:option('-netType',      'MSDNet', '')
   cmd:option('-retrain',      'none',   'Path to model to retrain with')
   cmd:option('-optimState',   'none',   'Path to an optimState to reload from')
   ---------- Model options ----------------------------------
   cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
   cmd:option('-optnet',          'false', 'Use optnet to reduce memory usage')
   ---------- MSDNet MOdel options ----------------------------------
   cmd:option('-base',           4,          'the layer to attach the first classifier')
   cmd:option('-nBlocks',        1,          'number of blocks/classifiers')
   cmd:option('-stepmode',       'even',     'patten of span between two adjacent classifers |even|lin_grow|')
   cmd:option('-step',           1,          'span between two adjacent classifers')
   cmd:option('-bottleneck',     'true',     'use 1x1 conv layer or not')
   cmd:option('-reduction',      0.5,        'dimension reduction ratio at transition layers')
   cmd:option('-growthRate',     6,          'number of output channels for each layer (the first scale)')
   cmd:option('-grFactor',       '1-2-4-4',  'growth rate factor of each sacle')
   cmd:option('-prune',          'max',      'specify how to prune the network, min | max')
   cmd:option('-joinType',       'concat',   'add or concat for features from different paths')
   cmd:option('-bnFactor',       '1-2-4-4',  'bottleneck factor of each sacle, 4-4-4-4 | 1-2-4-4')

   ---------- joint training options ----------------------------------
   cmd:option('-clearstate', 'true', 'save a model with clearsate or not')
   cmd:option('-joint_weight', 'uniform', 'weight of differnt classifiers: uniform | lin_grow | triangle | chi | gauss | exp')


   ---------- early exit testing options ----------------------------------
   cmd:option('-EEpath', 'none', 'the path to a saved model for early exit calculation')
   cmd:option('-EEensemble', 'true', 'use ensemble or not in early exit')

   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'
   opt.resetClassifier = opt.resetClassifier ~= 'false'
   opt.bottleneck = opt.bottleneck ~= 'false'
   opt.DataAug = opt.DataAug ~= 'false'
   opt.validset = opt.validset ~= 'false'
   opt.override = opt.override ~= 'false'
   opt.clearstate = opt.clearstate ~= 'false'
   opt.EEensemble = opt.EEensemble ~= 'false'

   -- for logging
   opt._grFactor = opt.grFactor
   opt._bnFactor = opt.bnFactor

   local bnFactor = {}
   for i, s in pairs(opt.bnFactor:split('-')) do
      bnFactor[i] = tonumber(s)
   end
   opt.bnFactor = bnFactor

   local grFactor = {}
   for i, s in pairs(opt.grFactor:split('-')) do
      grFactor[i] = tonumber(s)
   end
   opt.grFactor = grFactor

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end

   if opt.dataset == 'imagenet' then
      -- Handle the most common case of missing -data flag
      local trainDir = paths.concat(opt.data, 'train')
      if not paths.dirp(opt.data) then
         cmd:error('error: missing ImageNet data directory')
      elseif not paths.dirp(trainDir) then
         cmd:error('error: ImageNet missing `train` directory: ' .. trainDir)
      end
      -- Default shortcutType=B and nEpochs=90
      opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 90 or opt.nEpochs
      opt.nScales =  4
   elseif opt.dataset == 'cifar10' then
      -- Default shortcutType=A and nEpochs=164
      opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs
      opt.nScales =  3
   elseif opt.dataset == 'cifar100' then
       -- Default shortcutType=A and nEpochs=164
       opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
       opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs
       opt.nScales =  3
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if opt.precision == nil or opt.precision == 'single' then
      opt.tensorType = 'torch.CudaTensor'
   elseif opt.precision == 'double' then
      opt.tensorType = 'torch.CudaDoubleTensor'
   elseif opt.precision == 'half' then
      opt.tensorType = 'torch.CudaHalfTensor'
   else
      cmd:error('unknown precision: ' .. opt.precision)
   end

   if opt.shareGradInput and opt.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end

   return opt
end

return M
