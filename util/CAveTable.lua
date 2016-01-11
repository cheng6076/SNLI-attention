local CAveTable, parent = torch.class('nn.CAveTable', 'nn.Module')

function CAveTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

function CAveTable:updateOutput(input)
   self.output:resizeAs(input[1]):copy(input[1])
   for i=2,#input do
      self.output:add(input[i])
   end
   return self.output / (#input)
end

function CAveTable:updateGradInput(input, gradOutput)
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or input[1].new()
      self.gradInput[i]:resizeAs(input[i])
      self.gradInput[i]:copy(gradOutput / (#input))
   end
   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end

   return self.gradInput
end
