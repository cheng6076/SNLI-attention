--[1,0  +  [1,2  =  [2,2
-- 2,0]        ]     3,2

local TensorAddVector, parent = torch.class('nn.TensorAddVector', 'nn.Module')

function TensorAddVector:__init()
    parent.__init(self)
    self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function TensorAddVector:updateOutput(input)
    local vectors, scalars = unpack(input)
    self.output:resizeAs(vectors):copy(vectors):add(torch.repeatTensor(scalars, vectors:size(1)/scalars:size(1), 1))
    return self.output
end

function TensorAddVector:updateGradInput(input, gradOutput)
    local vectors, scalars = unpack(input)
    self.gradInput[1]:set(gradOutput)
    local tmp = gradOutput:clone():sum(1)
    self.gradInput[2]:set(tmp)
    return self.gradInput
end
