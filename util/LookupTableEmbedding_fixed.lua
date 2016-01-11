local LookupTableEmbedding_fixed, parent = torch.class('nn.LookupTableEmbedding_fixed', 'nn.Module')

function LookupTableEmbedding_fixed:__init(nIndex, nOutput, wordvec)
  parent.__init(self)
  self.weight = torch.Tensor(nIndex, nOutput)
  self.gradWeight = torch.Tensor(nIndex, nOutput):zero()
  self.output = torch.DoubleTensor()
  self:reset(wordvec)
end

function LookupTableEmbedding_fixed:reset(wordvec)
   self.weight:normal(0, 1)
   for key, value in pairs(wordvec) do
       self.weight[key] = value
   end
end

function LookupTableEmbedding_fixed:updateOutput(input)
  -- make sure input is a contiguous torch.LongTensor
  if (not input:isContiguous()) or torch.type(input) ~= 'torch.LongTensor' then
      self._indices = self._indices or torch.LongTensor()
      self._indices:resize(input:size()):copy(input)
      input = self._indices
  end
  if input:dim() == 1 then
      local nIndex = input:size(1)
      self.output:index(self.weight, 1, input)
  elseif input:dim() == 2 then
      -- batch mode
      local nExample = input:size(1)
      local nIndex = input:size(2)
      self._inputView = self._inputView or torch.LongTensor()
      self._inputView:view(input, -1)
      self.output:index(self.weight, 1, self._inputView)
      self.output = self.output:view(nExample, nIndex, self.weight:size(2))
  end
  return self.output
end


