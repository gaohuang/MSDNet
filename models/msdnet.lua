local nn = require 'nn'
require 'cunn'
require 'models/JointTrainContainer'

local function createModel(opt)
   -- (1) configure
   if opt.stepmode == 'even' then
      assert(opt.base - opt.step >= 0, 'Base should not be smaller than step!')
   end

   local nChannels
   if opt.dataset == 'cifar10' or opt.dataset == 'cifar100' then
      nChannels = opt.initChannels>0 and opt.initChannels or 16
   elseif opt.dataset == 'imagenet' then
      nChannels = opt.initChannels>0 and opt.initChannels or 64
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   -- (2) build model
   print(' | MSDNet-Block' .. opt.nBlocks.. '-'..opt.step .. ' ' .. opt.dataset)
   local model = nn.Sequential()
   model:add(nn.JointTrainModule(nChannels, opt))

   -- (3) init model
   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end
   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:type(opt.tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   -- (4) save the network definition
   local file = io.open(paths.concat(opt.save, 'model_definition.txt'), "w")
   for k, v in pairs(opt) do
      local s
      if v == true then
         s = 'true'
      elseif v == false then
         s = 'false'
      else
         s = v
      end
      file:write(tostring(k) .. ': '..tostring(s)..'\n')
   end

   file:write('\n model definition \n\n')
   file:write(model:__tostring__())
   file:close()

   print(model)

   return model
end

return createModel
