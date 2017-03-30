local save2txt = function(filename, data, precision)

   local precision = precision or 4

   local sz
   if torch.type(data) == 'table' and #data > 0 then
      local n = #data
      local d = #data[1]
      if torch.type(d) ~= 'number' then
         d = d[1]
      end
      sz = {n,d}
   elseif torch.isTensor(data) then
      sz = data:size()
   else
      print('Input cannot be saved to TXT file')
   end

   if #sz == 1 then
      local file = io.open(filename..'.txt', "w")
      for i = 1, sz[1] do
         file:write(string.format('%0.'..precision..'f\t', data[i]))
         file:write('\n')
      end
      file:close()
   elseif #sz ==2 then
      local file = io.open(filename..'.txt', "w")
      for i = 1, sz[1] do
         for j = 1, sz[2] do
            file:write(string.format('%0.'..precision..'f\t', data[i][j]))
         end
         file:write('\n')
      end
      file:close()
   else
      print('Input cannot be saved to TXT file')
   end
end

return save2txt
