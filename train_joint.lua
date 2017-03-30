
local optim = require 'optim'

local M = {}
local Trainer = torch.class('MSDNet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local lossSum = 0.0
   local N = 0

   local top1All, top5All = torch.zeros(self.opt.nBlocks), torch.zeros(self.opt.nBlocks)
   local top1Evolve, top5Evolve = torch.zeros(self.opt.nBlocks), torch.zeros(self.opt.nBlocks)

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input)
      local batchSize = output[1]:size(1)

      -- create a table which contains `nBlocks' same targets
      local multi_targets = {}
      for i = 1, self.opt.nBlocks do
         multi_targets[i] = self.target
      end
      local loss = self.criterion:forward(self.model.output, multi_targets)
      lossSum = lossSum + loss

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, multi_targets)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      -- sotre the cumulative softmax() of exits to obtain the ensemble performance
      local ensemble = torch.Tensor():resizeAs(output[1]:float()):zero()
      local top1, top5 = 0, 0
      for i = 1, self.opt.nBlocks do
         -- single exit
         top1, top5 = self:computeScore(output[i]:float(), sample.target, 1)
         top1All[i] = top1All[i] + top1*batchSize
         top5All[i] = top5All[i] + top5*batchSize
         -- ensemble
         ensemble:add(nn.SoftMax():forward(output[i]:float()))
         top1, top5 = self:computeScore(ensemble, sample.target, 1)
         top1Evolve[i] = top1Evolve[i] + top1*batchSize
         top5Evolve[i] = top5Evolve[i] + top5*batchSize
      end

      N = N + batchSize

      print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(
         epoch, n, trainSize, timer:time().real, dataTime, loss, top1All[self.opt.nBlocks]/N, top5All[self.opt.nBlocks]/N))

      -- check that the storage didn't get changed due to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   for i = 1, self.opt.nBlocks do
      top1All[i] = top1All[i] / N
      top5All[i] = top5All[i] / N
      top1Evolve[i] = top1Evolve[i] / N
      top5Evolve[i] = top5Evolve[i] / N
      print((' * Train %d exit single top1: %7.3f  top5: %7.3f, \t Ensemble %d exit(s) top1: %7.3f  top5: %7.3f')
      :format(i, top1All[i], top5All[i], i, top1Evolve[i], top5Evolve[i]))
   end

   return top1All, top5All, top1Evolve, top5Evolve, lossSum / N
end

function Trainer:test(epoch, dataloader, prefix)
   -- Computes the top-1 and top-5 err on the validation/test set
   prefix = prefix or 'Test'
   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local N = 0

   local top1All, top5All = torch.zeros(self.opt.nBlocks), torch.zeros(self.opt.nBlocks)
   local top1Evolve, top5Evolve = torch.zeros(self.opt.nBlocks), torch.zeros(self.opt.nBlocks)

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input)
      local batchSize = output[1]:size(1) / nCrops
      -- create a table which contains `nBlocks' same targets
      local multi_targets = {}
      for i = 1, self.opt.nBlocks do
         multi_targets[i] = self.target
      end
      local loss = self.criterion:forward(self.model.output, multi_targets)

      -- sotre the cumulative softmax() of exits to obtain the ensemble performance
      local ensemble = torch.Tensor():resizeAs(output[1]:float()):zero()
      local top1, top5 = 0, 0
      for i = 1, self.opt.nBlocks do
         -- single exit
         ensemble = ensemble or torch.Tensor():resizeAs(output[1]:float()):zero()
         top1, top5 = self:computeScore(output[i]:float(), sample.target, 1)
         top1All[i] = top1All[i] + top1*batchSize
         top5All[i] = top5All[i] + top5*batchSize
         -- ensemble
         ensemble:add(nn.SoftMax():forward(output[i]:float()))
         top1, top5 = self:computeScore(ensemble, sample.target, 1)
         top1Evolve[i] = top1Evolve[i] + top1*batchSize
         top5Evolve[i] = top5Evolve[i] + top5*batchSize
      end
      ----------------------------------------------------
      N = N + batchSize

      print((' | %s: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (cumul: %7.3f)  top5 %7.3f (cumul: %7.3f)'):format(
         prefix, epoch, n, size, timer:time().real, dataTime, top1, top1All[self.opt.nBlocks]/N,
         top5, top5All[self.opt.nBlocks]/N))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   for i = 1, self.opt.nBlocks do
      top1All[i] = top1All[i] / N
      top5All[i] = top5All[i] / N
      top1Evolve[i] = top1Evolve[i] / N
      top5Evolve[i] = top5Evolve[i] / N
      print((' * %s %d exit top1: %7.3f  top5: %7.3f, \t Ensemble %d exit(s) top1: %7.3f  top5: %7.3f'):format(
         prefix, i, top1All[i], top5All[i], i, top1Evolve[i], top5Evolve[i]))
   end

   return top1All, top5All, top1Evolve, top5Evolve
end

function Trainer:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _, predictions = output:float():topk(5, 2, true, true) -- descending sort

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(predictions))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

local function getCudaTensorType(tensorType)
   if tensorType == 'torch.CudaHalfTensor' then
     return cutorch.createCudaHostHalfTensor()
   elseif tensorType == 'torch.CudaDoubleTensor' then
    return cutorch.createCudaHostDoubleTensor()
   else
     return cutorch.createCudaHostTensor()
   end
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch[self.opt.tensorType:match('torch.(%a+)')]()
      or getCudaTensorType(self.opt.tensorType))
   self.target = self.target or (torch.CudaLongTensor and torch.CudaLongTensor())
   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 0.75*self.opt.nEpochs and 2 or epoch >= 0.5*self.opt.nEpochs and 1 or 0
   elseif self.opt.dataset == 'cifar100' then
      decay = epoch >= 0.75*self.opt.nEpochs and 2 or epoch >= 0.5*self.opt.nEpochs and 1 or 0
   end
   return self.opt.LR * math.pow(0.1, decay)
end


return M.Trainer