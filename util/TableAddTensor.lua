
local TableAddTensor, parent = torch.class('nn.TableAddTensor', 'nn.Module')

function TableAddTensor:__init()
   parent.__init(self)
   self.gradInput = {}
   self.output = {}
end

function TableAddTensor:updateOutput(input)
   self.output = {}
   local itable, tensor = unpack(input)
   for i=1, #itable do
     local output = torch.CudaTensor():resizeAs(tensor):copy(tensor)
     output:add(itable[i])   
     table.insert(self.output, output)   
   end
   return self.output
end

function TableAddTensor:updateGradInput(input, gradOutput)
   self.gradInput = {}
   local itable, tensor = unpack(input)
   local gradtable = {}
   local gradtensor = torch.CudaTensor():resizeAs(tensor):zero()
   for i=1, #itable do
      gradtensor:add(gradOutput[i])
      local gradInput = torch.CudaTensor():resizeAs(tensor):copy(gradOutput[i]) 
      table.insert(gradtable, gradInput)  
   end
   table.insert(self.gradInput, gradtable)   
   table.insert(self.gradInput, gradtensor)   

   return self.gradInput
end
