local SumKeepdim, parent = torch.class('nn.SumKeepdim', 'nn.Module')

function SumKeepdim:__init(dimension, nInputDims, sizeAverage)
   parent.__init(self)
   self.dimension   = dimension or 1
   self.nInputDims  = nInputDims
   self.sizeAverage = sizeAverage or false
end

function SumKeepdim:_getPositiveDimension(input)
    local dimension = self.dimension
    if dimension < 0 then
        dimension = input:dim() + dimension + 1
    elseif self.nInputDims and input:dim()==(self.nInputDims+1) then
        dimension = dimension + 1
    end
    assert(input:dim() >= dimension, "dimension exceeds input dimensions")
    return dimension
end

function SumKeepdim:updateOutput(input)
    local dimension = self:_getPositiveDimension(input)
    if type(self.output) == 'number' then
        self.output = input.new()
    end
    self.output:sum(input, dimension)
    if self.sizeAverage then
        self.output:div(input:size(dimension))
    end
    return self.output
end

function SumKeepdim:updateGradInput(input, gradOutput)
    local dimension = self:_getPositiveDimension(input)
    if not gradOutput:isContiguous() then
        self._gradOutput = self._gradOutput or gradOutput.new()
                self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
        gradOutput = self._gradOutput
    end
    self.gradInput:resizeAs(input)
    self.gradInput:copy(gradOutput)
    if self.sizeAverage then
        self.gradInput:div(input:size(dimension))
    end
    return self.gradInput
end

function SumKeepdim:clearState()
    nn.utils.clear(self, '_gradOutput')
    return parent.clearState(self)
end

