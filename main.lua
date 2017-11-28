--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'sys'
require 'paths'

local save2txt = require 'saveTXT'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train_joint'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model = models.setup(opt, checkpoint)

-- Use parallel criterion for multiple exits
local criterion = nn.ParallelCriterion()
for i = 1, opt.nBlocks do
   criterion:add(nn.CrossEntropyCriterion():type(opt.tensorType), 1)
end

-- Data loading
print('Creating dataloader ...')
local trainLoader, valLoader, testLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

-- Test only, need to specify -save and -retrain
if opt.testOnly then
   -- local flops = torch.load(paths.concat(opt.save, 'flops.t7'))
   -- opt.nBlocks = #flops
   local top1ErrValid, top5Err, top1ErrEnsembleValid, top5ErrEnsemble = trainer:test(0, valLoader)
   local top1Err, top5Err, top1ErrEnsemble, top5ErrEnsemble = trainer:test(0, testLoader)
   print('results from: ' .. opt.save)
   print(
      -- 'flops: \n', flops,
         '\nval single: \n', top1ErrValid,
         '\n val ensemble: \n', top1ErrEnsembleValid,
         '\n test single:\n', top1Err,
         '\n test ensemble:\n', top1ErrEnsemble)
   torch.save(paths.concat(opt.save, 'anytime_result.t7'), {top1ErrValid, top1ErrEnsembleValid,
                                                            top1Err, top1ErrEnsemble})
   return
end

-- Initialize some parameters
local paramsize = trainer.params:size(1)
print('Parameters:', paramsize)
checkpoints.save(0, model, trainer.optimState, false, opt)
local valSingle, valEnsemble, testSingle, testEnsemble = {}, {}, {}, {}
local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge
local timer = torch.Timer()

-- Training epochs
for epoch = startEpoch, opt.nEpochs do

   -- Train for a single epoch
   timer:reset()
   trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local valTop1All, _, valTop1Evolve = trainer:test(epoch, valLoader)
   valSingle[epoch] = valTop1All
   valEnsemble[epoch] = valTop1Evolve

   -- Run model on test set
   if opt.validset == true then
      local testTop1All, _, testTop1Evolve = trainer:test(epoch, testLoader)
      testSingle[epoch] = testTop1All
      testEnsemble[epoch] = testTop1Evolve
   end

   -- Log results to text file
   local filename = paths.concat(opt.save, 'result_')
   save2txt(filename..'valSingle', valSingle)
   save2txt(filename..'valEnsemble', valEnsemble)
   if opt.validset == true then
      save2txt(filename..'testSingle', testSingle)
      save2txt(filename..'testEnsemble', testEnsemble)
   end

   -- Checkpoint best model
   local bestModel = false
   if valEnsemble[epoch][opt.nBlocks] < bestTop1 then
      bestModel = true
      bestTop1 = valEnsemble[epoch][opt.nBlocks]
      print(' * Best model ', valEnsemble[epoch][opt.nBlocks])
      torch.save(paths.concat(opt.save, 'best_result_ensemble.t7'), {valEnsemble[epoch], testEnsemble[epoch]})
   end
   if bestModel or epoch == opt.nEpochs or opt.dataset == 'imagenet' then
      checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
   end
end

-- Done
print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))
