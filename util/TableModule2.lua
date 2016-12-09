local TableModule2, parent = torch.class('nn.TableModule2', 'nn.Module')

function TableModule2:__init(module)
   parent.__init(self)
   self.module = module
   self.output = {}
   self.gradInput = {}
end

function TableModule2:updateOutput(input)
   local input1, input2 = unpack(input)
   for i=1, #input1 do
     local tmp = self.module:updateOutput({input1[i], input2}) 
     table.insert(self.output, tmp)
   end
   return self.output
end
    
function TableModule2:updateGradInput(input, gradOutput)
   local input1, input2 = unpack(input)
   local gradInput1 = {}
   local gradInput2 
   for i=1, #input1 do 
     local tmp1, tmp2 = unpack(self.module:updateGradInput({input1[i],input2}, gradOutput[i]))
     table.insert(gradInput1, tmp1)
     if i==1 then
        gradInput2 = tmp2:clone()
     else
        gradInput2:add(tmp2)
     end
   end
   table.insert(self.gradInput, gradInput1)
   table.insert(self.gradInput, gradInput2)
   return self.gradInput
end
