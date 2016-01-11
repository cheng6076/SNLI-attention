local CWeightedTable, parent = torch.class('nn.CWeightedTable', 'nn.Module')

function CWeightedTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

function CWeightedTable:updateOutput(input)
   local total = (1+ (#input))* (#input) / 2
   self.output:resizeAs(input[1]):copy(input[1])
   for i=2,#input do
      self.output:add(input[i]*i)
   end
   return self.output / total
end

function CWeightedTable:updateGradInput(input, gradOutput)
   local total = (1+ (#input))* (#input) / 2
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or input[1].new()
      self.gradInput[i]:resizeAs(input[i])
      self.gradInput[i]:copy(gradOutput * i / total)
   end

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end

   return self.gradInput
end
