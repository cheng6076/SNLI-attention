local WeightedLookupTable, parent = torch.class('nn.WeightedLookupTable', 'nn.Module')

function WeightedLookupTable:__init(nIndex, nOutput)
   parent.__init(self)
   self.weight = torch.Tensor(nIndex, nOutput)
   self.gradWeight = torch.Tensor(nIndex, nOutput):zero()
   self:reset()
end

function WeightedLookupTable:reset(stdv)
   stdv = stdv or 1
   self.weight:normal(0, stdv)
end

function WeightedLookupTable:makeContiguous(x)
   if (not x:isContiguous())  then
      _x = torch.Tensor(x:size())
      _x:copy(x)
      return _x
   end
   return x
end

function WeightedLookupTable:updateOutput(input)
   if input:dim()==3  then
      local w = self.weight:view(1, self.weight:size()[1], self.weight:size()[2]):expand(input:size()[1], self.weight:size()[1], self.weight:size()[2])
      self.output = torch.bmm(input, w)
      return self.output
   elseif input:dim()==2 then
      self.output = torch.mm(input, self.weight)
      return self.output
   end
end

function WeightedLookupTable:updateGradInput(input, gradOutput)
   if self.gradInput then
      if input:dim()==3 then
         local weight = self:makeContiguous(self.weight:t())
         local w_p = weight:view(1, weight:size()[1], weight:size()[2]):expand(input:size()[1], weight:size()[1], weight:size()[2])
         self.gradInput = torch.bmm(gradOutput, w_p)
         return self.gradInput
      elseif input:dim()==2  then
         self.gradInput = torch.mm(gradOutput, self.weight:t())
         return self.gradInput
      end
   end 
end




