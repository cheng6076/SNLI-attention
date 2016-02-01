
-- misc utilities

function clone_list(tensor_list, zero_too)
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

function narrow_list(tensor_list, first, last, zero_too)
    local out = {}
    first = first or 1
    last = last or #tensor_list
    for i = first, last do
        if zero_too then
            table.insert(out, tensor_list[i]:clone():zero())
        else 
            table.insert(out, tensor_list[i])
        end
    end
    return out
end
