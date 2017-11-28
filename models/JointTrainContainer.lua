require 'nn'
require 'cudnn'
require 'cunn'
require 'models/MSDNet_Layer'


local function build_transition(nIn, nOut, outScales, offset, opt)
	local function conv1x1(nIn, nOut)
		local s = nn.Sequential()			
		s:add(cudnn.SpatialConvolution(nIn, nOut, 1,1, 1,1, 0,0))
		s:add(cudnn.SpatialBatchNormalization(nOut))
		s:add(cudnn.ReLU(true))
		return s	
	end
	local net = nn.ParallelTable()
	for i = 1, outScales do
		net:add(conv1x1(nIn * opt.grFactor[offset + i], nOut * opt.grFactor[offset + i]))
	end
	return net
end


local function build_block(nChannels, opt, step, layer_all, layer_tillnow)
   local block = nn.Sequential()
   if layer_tillnow == 0 then -- layer_curr == 0 means we are at the first block
		block:add(nn.MSDNet_Layer_first(3, nChannels, opt))
	end

   local nIn = nChannels
   for i = 1, step do
      local inScales, outScales = opt.nScales, opt.nScales
      layer_tillnow = layer_tillnow + 1
      if opt.prune == 'min' then
         inScales = math.min(opt.nScales, layer_all - layer_tillnow + 2)
         outScales = math.min(opt.nScales, layer_all - layer_tillnow + 1)
      elseif opt.prune == 'max' then
         local interval = torch.ceil(layer_all/opt.nScales)
         inScales = opt.nScales - torch.floor((math.max(0, layer_tillnow -2))/interval)
         outScales = opt.nScales - torch.floor((layer_tillnow -1)/interval)
      else
         error('Unknown prune option!')
      end
      print('|', 'inScales ', inScales, 'outScales ', outScales , '|')
      block:add(nn.MSDNet_Layer(nIn, opt.growthRate, opt, inScales, outScales))
      nIn = nIn + opt.growthRate
      if opt.prune == 'max' and inScales > outScales and opt.reduction > 0 then
         local offset = opt.nScales - outScales
         block:add(build_transition(nIn, math.floor(opt.reduction*nIn), outScales, offset, opt))
         nIn = math.floor(opt.reduction*nIn)
			print('|', 'Transition layer inserted!', '\t\t|')
      elseif opt.prune == 'min' and opt.reduction > 0 and (layer_tillnow == torch.floor(layer_all/3) or layer_tillnow == torch.floor(2*layer_all/3)) then
         local offset = opt.nScales - outScales
         block:add(build_transition(nIn, math.floor(opt.reduction*nIn), outScales, offset, opt))
         nIn = math.floor(opt.reduction*nIn)
         print('|', 'Transition layer inserted!', '\t\t|')
      end
   end
   return block, nIn
end

local function build_classifier_cifar(nChannels, nClass)
	local interChannels1, interChannels2 = 128, 128
	local c = nn.Sequential()
	c:add(cudnn.SpatialConvolution(nChannels, interChannels1,3,3,2,2,1,1))
	c:add(cudnn.SpatialBatchNormalization(interChannels1))
	c:add(cudnn.ReLU(true))
	c:add(cudnn.SpatialConvolution(interChannels1, interChannels2,3,3,2,2,1,1))
	c:add(cudnn.SpatialBatchNormalization(interChannels2))
	c:add(cudnn.ReLU(true))
	c:add(cudnn.SpatialAveragePooling(2,2))
	c:add(nn.Reshape(interChannels2))
	c:add(nn.Linear(interChannels2, nClass))
	return c
end

local function build_classifier_imagenet(nChannels, nClass)
	local c = nn.Sequential()
	c:add(cudnn.SpatialConvolution(nChannels, nChannels,3,3,2,2,1,1))
	c:add(cudnn.SpatialBatchNormalization(nChannels))
	c:add(cudnn.ReLU(true))
	c:add(cudnn.SpatialConvolution(nChannels, nChannels,3,3,2,2,1,1))
	c:add(cudnn.SpatialBatchNormalization(nChannels))
	c:add(cudnn.ReLU(true))
	c:add(cudnn.SpatialAveragePooling(2,2))
	c:add(nn.Reshape(nChannels))
	c:add(nn.Linear(nChannels, 1000))
	return c
end

local JointTrainModule, parent = torch.class('nn.JointTrainModule', 'nn.Container')

function JointTrainModule:__init(nChannels, opt)
	parent.__init(self)

	self.train = true
	self.nChannels = nChannels
	self.nBlocks = opt.nBlocks
	self.opt = opt

	local nIn = nChannels
	self.modules = {}

   -- calculate the step size of each blocks
   local layer_curr, layer_all = 0, opt.base
   local steps = {}
   steps[1] = opt.base
   for i = 2, self.nBlocks do
	  steps[i] = opt.stepmode=='even' and opt.step or opt.stepmode=='lin_grow' and opt.step*(i-1)+1
      layer_all = layer_all + steps[i]
   end
   print("building network of steps: ")
   print(steps)
   torch.save(paths.concat(opt.save, 'layer_specific.t7'), steps)

	for i = 1, self.nBlocks do
		print(' ----------------------- Block ' .. i .. ' -----------------------')
		self.modules[i], nIn = build_block(nIn, opt, steps[i], layer_all, layer_curr)
      layer_curr = layer_curr + steps[i]
		if opt.dataset == 'cifar10' then
			self.modules[i+self.nBlocks] = build_classifier_cifar(nIn*opt.grFactor[opt.nScales], 10)
		elseif opt.dataset == 'cifar100' then
			self.modules[i+self.nBlocks] = build_classifier_cifar(nIn*opt.grFactor[opt.nScales], 100)
		elseif opt.dataset == 'imagenet' then
			self.modules[i+self.nBlocks] = build_classifier_imagenet(nIn*opt.grFactor[opt.nScales], 1000)
		else
			error('Unknown dataset!')
		end
   end
   self.gradInput = {}
   self.output = {}

end

function JointTrainModule:updateOutput(input)
	for i = 1, self.nBlocks do
		local tmp_input = (i==1) and input or self.modules[i-1].output
		local tmp_output1 = self:rethrowErrors(self.modules[i], i, 'updateOutput', tmp_input)
		local tmp_output2 = self:rethrowErrors(self.modules[i+self.nBlocks], i+self.nBlocks, 'updateOutput', tmp_output1[#tmp_output1])
		self.output[i] = tmp_output2
	end
	return self.output
end

function JointTrainModule:updateGradInput(input, gradOutput)

local nScales = self.opt.nScales
	for i = self.nBlocks, 1, -1 do
		local features = self.modules[i].output[#self.modules[i].output]
		self.modules[i+self.nBlocks]:updateGradInput(features, gradOutput[i])
		local gOut = {}
		if i == self.nBlocks then
			for s = 1, nScales do
				local out = self.modules[i].output[s]
				if out then
					gOut[s] = out.new():resizeAs(out):zero()
				end
			end
		else
			gOut = self.modules[i+1].gradInput
		end
		gOut[#gOut]:add(self.modules[i+self.nBlocks].gradInput)
		local tmp_input = (i==1) and input or self.modules[i-1].output 
		self.modules[i]:updateGradInput(tmp_input, gOut)
	end
self.gradInput = self.modules[1].gradInput

return self.gradInput
end

function JointTrainModule:accGradParameters(input, gradOutput, scale)
	scale = scale or 1
	for i = self.nBlocks, 1, -1 do
		local features = self.modules[i].output[#self.modules[i].output]
		self.modules[i+self.nBlocks]:accGradParameters(features, gradOutput[i], scale)
		local tmp_input = (i==1) and input or self.modules[i-1].output
		local gOut
		if i == self.nBlocks then
			gOut = self.modules[i].output
		else
			gOut = self.modules[i+1].gradInput
		end
		self.modules[i]:accGradParameters(tmp_input, gOut, scale)
end
end

function JointTrainModule:__tostring__()
	local tab = '  '
	local line = '\n'
	local next = '  |`-> '
	local lastNext = '   `-> '
	local ext = '  |    '
	local extlast = '       '
	local last = '   ... -> '
	local str = 'JointTrainModule'
	str = str .. ' {' .. line .. tab .. '{input}'
	for i=1,#self.modules do
	  if i == #self.modules then
	     str = str .. line .. tab .. lastNext .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
	  else
	     str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
	  end
	end
	str = str .. line .. tab .. last .. '{output}'
	str = str .. line .. '}'
	return str
end

