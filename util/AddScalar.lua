--[1,0,2,3  +  [1,2  =  [2,2,3,5
-- 2,0,1,3]    [2,3]     4,3,3,6]

local AddScalar, parent = torch.class('nn.AddScalar', 'nn.Module')

function AddScalar:__init()
    parent.__init(self)
    self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function AddScalar:updateOutput(input)
    local vectors, scalars = unpack(input)
    self.output:resizeAs(vectors):copy(vectors):add(torch.repeatTensor(scalars, 1, vectors:size(2)/scalars:size(2)))
    return self.output
end

function AddScalar:updateGradInput(input, gradOutput)
    local vectors, scalars = unpack(input)
    self.gradInput[1]:set(gradOutput)
    local tmp = gradOutput:clone():view(scalars:size(1), -1, scalars:size(2)):sum(2):squeeze()
    self.gradInput[2]:set(tmp)
    return self.gradInput
end
