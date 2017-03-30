--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This automatically downloads the CIFAR-10 dataset from
--  http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz
--

local URL = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'

local M = {}

local function convertToTensor(files)
   local data, labels

   for _, file in ipairs(files) do
      local m = torch.load(file, 'ascii')
      if not data then
         data = m.data:t()
         labels = m.labels:squeeze()
      else
         data = torch.cat(data, m.data:t(), 1)
         labels = torch.cat(labels, m.labels:squeeze())
      end
   end

   -- This is *very* important. The downloaded files have labels 0-9, which do
   -- not work with CrossEntropyCriterion
   labels:add(1)
   
   return {
      data = data:contiguous():view(-1, 3, 32, 32),
      labels = labels,
   }
end

function M.exec(opt, cacheFile)
   local rawpath = 'gen/cifar-10-batches-t7/data_batch_1.t7'
   if not paths.filep(rawpath) then
      print("=> Downloading CIFAR-10 dataset from " .. URL)
      local ok = os.execute('curl ' .. URL .. ' | tar xz -C gen/')
      assert(ok == true or ok == 0, 'error downloading CIFAR-10')
   end

   print(" | combining dataset into a single file")
   local trainData = convertToTensor({
      'gen/cifar-10-batches-t7/data_batch_1.t7',
      'gen/cifar-10-batches-t7/data_batch_2.t7',
      'gen/cifar-10-batches-t7/data_batch_3.t7',
      'gen/cifar-10-batches-t7/data_batch_4.t7',
      'gen/cifar-10-batches-t7/data_batch_5.t7',
   })
   local testData = convertToTensor({
      'gen/cifar-10-batches-t7/test_batch.t7',
   })
   print(" | saving CIFAR-10 dataset to " .. cacheFile)


   if opt.validset == true then
      torch.manualSeed(1)
      local shuffle = torch.randperm(50000)
      trainData.data[{ {1, 50000} }] = trainData.data:index(1, shuffle:long())
      trainData.labels[{ {1, 50000} }] = trainData.labels:index(1, shuffle:long())

      local valData = {data = torch.Tensor(), labels = torch.Tensor()}
      valData.data  = trainData.data[{ {45001,50000} }]
      valData.labels  =  trainData.labels[{ {45001,50000} }]
      trainData.data = trainData.data[{ {1,45000} }]
      trainData.labels = trainData.labels[{ {1,45000} }]

      torch.save(cacheFile, {
         train = trainData,
         val = valData,
         test = testData,})
   else
      torch.save(cacheFile, {
         train = trainData,
         val = testData,})
   end

end

return M
