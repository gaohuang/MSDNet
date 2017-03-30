require 'nn'
require 'cudnn'
require 'cunn'

local function build(nChannels, nOutChannels, type, bottleneck, bnWidth)
   local net = nn.Sequential()
   local innerChannels = nChannels
   bnWidth = bnWidth or 4
   if bottleneck == true then
      innerChannels = math.min(innerChannels, bnWidth * nOutChannels)
      net:add(cudnn.SpatialConvolution(nChannels, innerChannels, 1,1, 1,1, 0,0))
      net:add(cudnn.SpatialBatchNormalization(innerChannels))
      net:add(cudnn.ReLU(true))
   end
   if type == 'normal' then
      net:add(cudnn.SpatialConvolution(innerChannels, nOutChannels, 3,3, 1,1, 1,1))
   elseif type == 'down' then
      net:add(cudnn.SpatialConvolution(innerChannels, nOutChannels, 3,3, 2,2, 1,1))
   elseif type == 'up' then
      net:add(cudnn.SpatialFullConvolution(innerChannels, nOutChannels, 3,3, 2,2, 1,1, 1,1))
   else
      error("Please implement me: " .. type)
   end
   net:add(cudnn.SpatialBatchNormalization(nOutChannels))
   net:add(cudnn.ReLU(true))
   return net
end

local function build_net_normal(nChannels, nOutChannels, bottleneck, bnWidth)
   local net_warp = nn.Sequential()
   local net = nn.ParallelTable()
   net:add(nn.Identity())
   net:add(build(nChannels, nOutChannels, 'normal', bottleneck, bnWidth))
   net_warp:add(net):add(nn.JoinTable(2))
   return net_warp
end

local function build_net_down_normal(nChannels1, nChannels2, nOutChannels, bottleneck, bnWidth1, bnWidth2)
   local net_warpper = nn.Sequential()
   local net = nn.ParallelTable()
   assert(nOutChannels % 2 == 0, 'Growth rate invalid!')
   net:add(nn.Identity())
   net:add(build(nChannels1, nOutChannels/2, 'down', bottleneck, bnWidth1))
   net:add(build(nChannels2, nOutChannels/2, 'normal', bottleneck, bnWidth2))
   net_warpper:add(net):add(nn.JoinTable(2))
   return net_warpper
end


----------------
--- MSDNet_Layer_first:
---      the input layer of MSDNet
--- input: a tensor (orginal image)
--- output: a table of nScale tensors
----------------

local MSDNet_Layer_first, parent = torch.class('nn.MSDNet_Layer_first', 'nn.Container')

function MSDNet_Layer_first:__init(nChannels, nOutChannels, opt)
   parent.__init(self)

   self.train = true
   self.nChannels = nChannels
   self.nOutChannels = nOutChannels

   self.opt = opt

   self.modules = {}
   -- transform raw input to first layer
   self.modules[1] = nn.Sequential()
   if opt.dataset == 'cifar10' or opt.dataset == 'cifar100' then
      self.modules[1]:add(cudnn.SpatialConvolution(nChannels, nOutChannels*opt.grFactor[1], 3,3, 1,1, 1,1))
      self.modules[1]:add(cudnn.SpatialBatchNormalization(nOutChannels*opt.grFactor[1]))
      self.modules[1]:add(cudnn.ReLU(true))
   elseif opt.dataset == 'imagenet' then
      self.modules[1]:add(cudnn.SpatialConvolution(nChannels,nOutChannels*opt.grFactor[1], 7,7, 2,2, 3,3))
      self.modules[1]:add(nn.SpatialBatchNormalization(nOutChannels*opt.grFactor[1]))
      self.modules[1]:add(cudnn.ReLU(true))
      self.modules[1]:add(nn.SpatialMaxPooling(3,3,2,2,1,1))
   end

   local nIn = nOutChannels * opt.grFactor[1]
   for i = 2, opt.nScales do
      self.modules[i] = nn.Sequential()
      self.modules[i]:add(cudnn.SpatialConvolution(nIn, nOutChannels*opt.grFactor[i], 3,3, 2,2, 1,1))
      self.modules[i]:add(cudnn.SpatialBatchNormalization(nOutChannels*opt.grFactor[i]))
      self.modules[i]:add(cudnn.ReLU(true))
      nIn = nOutChannels*opt.grFactor[i]
   end

   self.gradInput = torch.CudaTensor()
   self.output = {}
end

function MSDNet_Layer_first:updateOutput(input)
   for i = 1, self.opt.nScales do
      self.output[i] = self.output[i] or input.new()
      local tmp_input = (i==1) and input or self.output[i-1]
      local tmp_output = self:rethrowErrors(self.modules[i], i, 'updateOutput', tmp_input)
      self.output[i]:resizeAs(tmp_output):copy(tmp_output)
   end
   return self.output
end

function MSDNet_Layer_first:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or input.new()
   self.gradInput:resizeAs(input):zero()
   for i = self.opt.nScales-1, 1, -1 do
      gradOutput[i]:add(self.modules[i+1]:updateGradInput(self.output[i], gradOutput[i+1]))
   end
   self.gradInput:resizeAs(input):copy(self.modules[1]:updateGradInput(input, gradOutput[1]))
   return self.gradInput
end

function MSDNet_Layer_first:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   for i = self.opt.nScales, 2, -1 do
      self.modules[i]:accGradParameters(self.output[i-1], gradOutput[i], scale)
   end
   self.modules[1]:accGradParameters(input, gradOutput[1], scale)
end

function MSDNet_Layer_first:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local lastNext = '   `-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = 'MSDNet_Layer_first'
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

----------------
--- MSDNet_Layer (without upsampling):
---      subsequent layers of MSDNet
--- input: a table of `nScales` tensors
--- output: a table of `nScales` tensors
----------------


local MSDNet_Layer, parent = torch.class('nn.MSDNet_Layer', 'nn.Container')


function MSDNet_Layer:__init(nChannels, nOutChannels, opt, inScales, outScales)
   parent.__init(self)

   self.train = true
   self.nChannels = nChannels
   self.nOutChannels = nOutChannels
   self.opt = opt
   self.inScales = inScales or opt.nScales
   self.outScales = outScales or opt.nScales
   self.nScales = opt.nScales
   self.discard = self.inScales - self.outScales
   assert(self.discard<=1, 'Double check inScales'..self.inScales..'and outScales: '..self.outScales)
   
   local offset = self.nScales - self.outScales
   self.modules = {}
   local isTrans = self.outScales<self.inScales
   if isTrans then
      local nIn1, nIn2, nOut = nChannels*opt.grFactor[offset], nChannels*opt.grFactor[offset+1], opt.grFactor[offset+1]*nOutChannels
      self.modules[1] = build_net_down_normal(nIn1, nIn2, nOut, opt.bottleneck, opt.bnFactor[offset], opt.bnFactor[offset+1])
   else
      self.modules[1] = build_net_normal(nChannels*opt.grFactor[offset+1], opt.grFactor[offset+1]*nOutChannels, opt.bottleneck, opt.bnFactor[offset+1])
   end
   for i = 2, self.outScales do
      local nIn1, nIn2, nOut = nChannels*opt.grFactor[offset+i-1], nChannels*opt.grFactor[offset+i], opt.grFactor[offset+i]*nOutChannels
      self.modules[i] = build_net_down_normal(nIn1, nIn2, nOut, opt.bottleneck, opt.bnFactor[offset+i-1], opt.bnFactor[offset+i])
   end

   self.real_input = {}
   self.gradInput = {}
   self.gIn = {}
   self.output = {}

   self.modules = self.modules

end


function MSDNet_Layer:updateOutput(input)


   local discard = self.discard
   if self.inScales == self.outScales then
      self.real_input[1] = {input[1], input[1]}
      for i = 2, self.outScales do
         self.real_input[i] = {input[i], input[i-1], input[i]}
      end
   else
      for i = 1, self.outScales do -- !!!
         self.real_input[i] = {input[discard+i], input[discard+i-1], input[discard+i]}
      end
   end

   for i = 1, self.outScales do
      self.output[i] = self:rethrowErrors(self.modules[i], i, 'updateOutput', self.real_input[i])
   end      

   return self.output
end


function MSDNet_Layer:updateGradInput(input, gradOutput)

   -- self.gradInput = self.gradInput or {}
   local offset = self.offset
   self.gIn = self.gIn or {}

   for i = 1, self.inScales do
      self.gradInput[i] = self.gradInput[i] or input[i].new()
   end

   for i = 1, self.outScales do
      self.gIn[i] = self.modules[i]:updateGradInput(self.real_input[i], gradOutput[i])
   end

   if self.inScales == self.outScales then
      if self.outScales == 1 then
         self.gradInput[1]:resizeAs(self.gIn[1][1]):copy(self.gIn[1][1])
                    :add(self.gIn[1][2])
      else
         self.gradInput[1]:resizeAs(self.gIn[1][1]):copy(self.gIn[1][1])
                             :add(self.gIn[1][2]):add(self.gIn[2][2])
      end
      for i = 2, self.inScales-1 do 
         self.gradInput[i]:resizeAs(self.gIn[i][1]):copy(self.gIn[i][1])
                       :add(self.gIn[i][3]):add(self.gIn[i+1][2])
      end

   else
      self.gradInput[1]:resizeAs(self.gIn[1][2]):copy(self.gIn[1][2])
      for i = 2, self.inScales - 1 do 
         self.gradInput[i]:resizeAs(self.gIn[i-1][1]):copy(self.gIn[i-1][1])
                       :add(self.gIn[i-1][3]):add(self.gIn[i][2])
      end 
   end
   if self.inScales > 1 then
      self.gradInput[self.inScales]:resizeAs(self.gIn[self.outScales][1])
      :copy(self.gIn[self.outScales][1]):add(self.gIn[self.outScales][3])
   end    

   return self.gradInput

end


function MSDNet_Layer:accGradParameters(input, gradOutput, scale)
   
   scale = scale or 1
   for i = 1, self.outScales do
      self.modules[i]:accGradParameters(self.real_input[i], gradOutput[i], scale)
   end
end


function MSDNet_Layer:clearState()
   -- don't call set because it might reset referenced tensors
   local function clear(f)
      if self[f] then
         if torch.isTensor(self[f]) then
            self[f] = self[f].new()
         elseif type(self[f]) == 'table' then
            self[f] = {}
         else
            self[f] = nil
         end
      end
   end
   clear('output')
   clear('gradInput')
   clear('real_input')
   clear('gIn')
   if self.modules then
      for i,module in pairs(self.modules) do
         module:clearState()
      end
   end
   return self
end

function MSDNet_Layer:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local lastNext = '   `-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = 'MSDNet_Layer'
   str = str .. ' {' .. line .. tab .. '{input}'
   for i = 1,#self.modules do
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