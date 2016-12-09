local THNN = require 'nn.THNN'
local LookupTableEmbedding_train, parent = torch.class('nn.LookupTableEmbedding_train', 'nn.Module')

function LookupTableEmbedding_train:__init(nIndex, nOutput, wordvec, paddingValue, maxNorm, normType)
   parent.__init(self)

   self.weight = torch.Tensor(nIndex, nOutput)
   self.gradWeight = torch.Tensor(nIndex, nOutput):zero()
   self.paddingValue = paddingValue or 0
   self.maxNorm = maxNorm or nil
   self.normType = normType or nil

   self:reset(wordvec)
end

function LookupTableEmbedding_train:backCompatibility()
   self._count = self._count or torch.IntTensor()
   self._input = self._input or torch.LongTensor()

   if not self.shouldScaleGradByFreq then
      self.shouldScaleGradByFreq = false
   end
end

function LookupTableEmbedding_train:accUpdateOnly()
   self.gradWeight = nil
   return self
end

function LookupTableEmbedding_train:setPadding(paddingValue)
   self.paddingValue = paddingValue
   return self
end

function LookupTableEmbedding_train:setMaxNorm(maxNorm)
   self.maxNorm = maxNorm
   return self
end

function LookupTableEmbedding_train:setNormType(normType)
   self.normType = normType
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
   if (not input:isContiguous()) or torch.type(input) ~= torch.type(self._input) then
      self.copiedInput = true
      self._input:resize(input:size()):copy(input)
      return self._input
   end
   self.copiedInput = false
   return input
end

function LookupTableEmbedding_train:updateOutput(input)
   self:backCompatibility()
   self:renorm(input)
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

function LookupTableEmbedding_train:updateGradInput(input, gradOutput)
   if torch.type(self.gradInput) ~= torch.type(input) then
      self.gradInput = input.new()
   end
   if not self.gradInput:isSameSizeAs(input) then
      self.gradInput:resizeAs(input):zero()
   end
   return self.gradInput
end

function LookupTableEmbedding_train:accGradParameters(input, gradOutput, scale)
   self:backCompatibility()
   input = self.copiedInput and self._input or input
   if input:dim() == 2 then
      input = input:view(-1)
   elseif input:dim() ~= 1 then
      error("input must be a vector or matrix")
   end

   if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end

   self.gradWeight.THNN.LookupTable_accGradParameters(
      input:cdata(),
      gradOutput:cdata(),
      self.gradWeight:cdata(),
      self._count:cdata(),
      THNN.optionalTensor(self._sorted),
      THNN.optionalTensor(self._indices),
      self.shouldScaleGradByFreq or false,
      self.paddingValue or 0,
      scale or 1
   )
end

function LookupTableEmbedding_train:renorm(input)
   if not self.maxNorm then
      return
   end
   self._input:resize(input:size()):copy(input)
   local row_idx = self._input
   if row_idx:dim() == 2 then
      row_idx = row_idx:view(-1)
   elseif row_idx:dim() ~= 1 then
      error("input must be a vector or matrix")
   end
   self.weight.THNN.LookupTableEmbedding_train_renorm(
      row_idx:cdata(),
      self.weight:cdata(),
      self.maxNorm,
      self.normType or 2
   )
end

function LookupTableEmbedding_train:type(type, tensorCache)
   parent.type(self, type, tensorCache)

   if type == 'torch.CudaTensor' then
      self._sorted = torch.CudaLongTensor.new()
      self._indices = torch.CudaLongTensor.new()
      self._count = torch.CudaLongTensor.new()
      self._input = torch.CudaLongTensor.new()
   else
      self._count = torch.IntTensor()
      self._input = torch.LongTensor()
   end

   return self
end

function LookupTableEmbedding_train:clearState()
   nn.utils.clear(self, '_count', '_input', '_gradOutput')
   return parent.clearState(self)
end

LookupTableEmbedding_train.sharedAccUpdateGradParameters = LookupTableEmbedding_train.accUpdateGradParameters

