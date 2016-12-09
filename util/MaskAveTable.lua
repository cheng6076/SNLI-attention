local MaskAveTable, parent = torch.class('nn.MaskAveTable', 'nn.Module')
function MaskAveTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

function MaskAveTable:updateOutput(input)
   local tmp, mask = unpack(input)
   -- Batch only
   self.output:resizeAs(tmp[1]):copy(tmp[1])
   for i=2,#tmp do
      self.output:add(tmp[i]:clone():cmul(mask[{{},i}]:contiguous():view(-1, 1):expandAs(tmp[i])))
   end
   local count = mask:sum(2):view(-1, 1):expandAs(self.output)
   self.output = self.output:cdiv(count) 
   return self.output
end

function MaskAveTable:updateGradInput(input, gradOutput)
   local tmp, mask = unpack(input)
   local count = mask:sum(2):view(-1, 1):expandAs(gradOutput)
   local total = gradOutput:clone():cdiv(count)
   self.gradInput[1] = {}
   self.gradInput[2] = mask:clone():zero()
   for i=1,#tmp do
      self.gradInput[1][i] = tmp[i]:clone():zero()
      self.gradInput[1][i]:copy(total:clone():cmul(mask[{{},i}]:contiguous():view(-1, 1):expandAs(tmp[i])))
   end
   for i=#tmp+1, #self.gradInput do
       self.gradInput[1][i] = nil
   end
   return self.gradInput
end
