local TableModule, parent = torch.class('nn.TableModule', 'nn.Module')

function TableModule:__init(module)
   parent.__init(self)
   self.module = module
   self.output = {}
   self.gradInput = {}
end

function TableModule:updateOutput(input)
   for i=1, #input do 
     table.insert(self.output, self.module:updateOutput(input[i]))
   end
   return self.output
end
    
function TableModule:updateGradInput(input, gradOutput)
   for i=1, #input do 
     table.insert(self.gradInput, self.module:updateGradInput(input[i], gradOutput[i]))
   end
   return self.gradInput
end
