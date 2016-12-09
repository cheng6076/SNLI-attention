local TableLinear, parent = torch.class('nn.TableLinear', 'nn.Module')

function TableLinear:__init(inputSize, outputSize)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)
   self.output = {}
   self.gradInput = {}
   self:reset()
end

function TableLinear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end

   return self
end

function TableLinear:updateOutput(input_table)
   self.output = {}
   for i=1, #input_table do
      if input_table[i]:dim() == 1 then
         local output = torch.CudaTensor():resize(self.bias:size(1))
         output:copy(self.bias)
         output:addmv(1, self.weight, input_table[i])
         table.insert(self.output, output)
      elseif input_table[i]:dim() == 2 then
         local nframe = input_table[i]:size(1)
         local output = torch.CudaTensor():resize(nframe, self.bias:size(1))
         if not self.addBuffer or self.addBuffer:size(1) ~= nframe then
            self.addBuffer = input_table[i].new(nframe):fill(1)
         end
         output:addmm(0, output, 1, input_table[i], self.weight:t())
         output:addr(1, self.addBuffer, self.bias)
         table.insert(self.output, output)
      else
         error('input must be vector or matrix')
      end
  end
  return self.output
end

function TableLinear:updateGradInput(input_table, gradOutput_table)
   self.gradInput = {}
   if self.gradInput then
      for i=1, #input_table  do 
         local gradInput=torch.CudaTensor():resizeAs(input_table[i])
         if input_table[i]:dim() == 1 then
            gradInput:addmv(0, 1, self.weight:t(), gradOutput_table[i])
         elseif input_table[i]:dim() == 2 then
            gradInput:addmm(0, 1, gradOutput_table[i], self.weight)
         end
         table.insert(self.gradInput, gradInput)
      end
      return self.gradInput
   end
end

function TableLinear:accGradParameters(input_table, gradOutput_table, scale)
   scale = scale or 1
   for i=1, #input_table do
      if input_table[i]:dim() == 1 then
         self.gradWeight:addr(scale, gradOutput_table[i], input_table[i])
         self.gradBias:add(scale, gradOutput_table[i])
      elseif input_table[i]:dim() == 2 then
         self.gradWeight:addmm(scale, gradOutput_table[i]:t(), input_table[i])
         self.gradBias:addmv(scale, gradOutput_table[i]:t(), self.addBuffer)
      end
   end
end

-- we do not need to accumulate parameters when sharing
TableLinear.sharedAccUpdateGradParameters = TableLinear.accUpdateGradParameters


function TableLinear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
