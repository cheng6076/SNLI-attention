local LookupTableEmbedding_train, parent = torch.class('nn.LookupTableEmbedding_train', 'nn.Module')

function LookupTableEmbedding_train:__init(nIndex, nOutput, wordvec)
   parent.__init(self)
   self.weight = torch.Tensor(nIndex, nOutput)
   self.gradWeight = torch.Tensor(nIndex, nOutput):zero()
   self._count = torch.IntTensor()
   self._input = torch.LongTensor()
   self.shouldScaleGradByFreq = false
   self.output = torch.DoubleTensor()
   self.wordvec = wordvec
   self:reset(self.wordvec)
end

function LookupTableEmbedding_train:accUpdateOnly()
   self.gradWeight = nil
   return self
end

function LookupTableEmbedding_train:scaleGradByFreq()
   self.shouldScaleGradByFreq = true
   return self
end

function LookupTableEmbedding_train:reset(wordvec)
   self.weight:normal(0, 1)
   for key, value in pairs(wordvec) do
       self.weight[key] = value
   end
end

function LookupTableEmbedding_train:makeInputContiguous(input)
   -- make sure input is a contiguous torch.LongTensor
   if (not input:isContiguous()) or torch.type(input) ~= torch.type(self._input) then
      self._input:resize(input:size()):copy(input)
      return self._input
   end
   return input
end

function LookupTableEmbedding_train:updateOutput(input)
   input = self:makeInputContiguous(input)
   if input:dim() == 1 then
      self.output:index(self.weight, 1, input)
   elseif input:dim() == 2 then
      self.output:index(self.weight, 1, input:view(-1))
      self.output = self.output:view(input:size(1), input:size(2), self.weight:size(2))
   else
      error("input must be a vector or matrix")
   end
   return self.output
end

function LookupTableEmbedding_train:accGradParameters(input, gradOutput, scale)
   input = self:makeInputContiguous(input)
   self.gradWeight.nn.LookupTable_accGradParameters(self, input, gradOutput, scale)
end

function LookupTableEmbedding_train:type(type)
   parent.type(self, type)

   if type == 'torch.CudaTensor' then
      -- CUDA uses _sorted and _indices temporary tensors
      self._sorted = self.weight.new()
      self._indices = self.weight.new()
   else
      -- self._count and self._input should only be converted if using Cuda
      self._count = torch.IntTensor()
      self._input = torch.LongTensor()
   end
   return self
end

-- we do not need to accumulate parameters when sharing
LookupTableEmbedding_train.sharedAccUpdateGradParameters = LookupTableEmbedding_train.accUpdateGradParameters








