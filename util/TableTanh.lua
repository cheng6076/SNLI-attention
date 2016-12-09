local TableTanh, parent = torch.class('nn.TableTanh', 'nn.Module')

function TableTanh:__init()
   parent.__init(self)
   self.output = {}
   self.gradInput = {}
   self.tanh = nn.Tanh()
end

function TableTanh:updateOutput(input)
   self.output = {}
   for i=1, #input do 
     table.insert(self.output, self.tanh:forward(input[i]))
   end
   return self.output
end
    
function TableTanh:updateGradInput(input, gradOutput)
   self.gradInput = {}
   for i=1, #input do 
     table.insert(self.gradInput, self.tanh:backward(input[i], gradOutput[i]))
   end
   return self.gradInput
end
