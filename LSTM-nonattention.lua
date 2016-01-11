require 'torch'
require 'nn'
require 'nngraph'
require 'rnn'
model_utils = require 'util.model_utils'
BatchLoader = require 'util.BatchLoader'
require 'util.misc'
require 'util.CAveTable'
classifier_simple = require 'model.classifier_simple'
encoder = require 'model.encoder'
decoder = require 'model.decoder'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-data_dir', 'data', 'path of the dataset')
cmd:option('-max_epochs', 10, 'number of full passes through the training data')
cmd:option('-rnn_size', 300, 'dimensionality of sentence embeddings')
cmd:option('-word_vec_size', 300, 'dimensionality of word embeddings')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-dropout',0.5,'dropout. 0 = no dropout')
cmd:option('-seed',3435,'torch manual random number generator seed')
cmd:option('-print_every',500,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every', 50000, 'save epoch')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','model','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-checkpoint', 'checkpoint.t7', 'start from a checkpoint if a valid checkpoint.t7 file is given')
cmd:option('-learningRate', 0.1, 'learning rate')
cmd:option('-decayRate',0.75,'decay rate for sgd')
cmd:option('-decay_when',0.1,'decay if validation does not improve by more than this much')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-max_grad_norm',5,'normalize gradients at')
-- GPU/CPU
cmd:option('-gpuid', -1,'which gpu to use. -1 = use CPU')
cmd:option('-cudnn', 0,'use cudnn (1=yes). this should greatly speed up convolutions')
cmd:option('-time', 0, 'print batch times')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- load necessary packages depending on config options
if opt.gpuid >= 0 then
   print('using CUDA on GPU ' .. opt.gpuid .. '...')
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.gpuid + 1)
end
if opt.cudnn == 1 then
  assert(opt.gpuid >= 0, 'GPU must be used if using cudnn')
  print('using cudnn...')
  require 'cudnn'
end

-- create data loader
loader = BatchLoader.create(opt.data_dir)
opt.seq_length = loader.max_sentence_l 
opt.vocab_size = #loader.idx2word
opt.classes = 3
-- model
protos = {}
protos.enc = encoder.lstm(opt.vocab_size, opt.rnn_size, opt.num_layers, opt.dropout, opt.word_vec_size)
protos.dec = decoder.lstm(opt.vocab_size, opt.rnn_size, opt.num_layers, opt.dropout, opt.word_vec_size)
protos.criterion = nn.ClassNLLCriterion()
protos.classifier = classifier_simple.classifier(opt.rnn_size, opt.dropout, opt.classes)
-- ship to gpu
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end
-- params and grads
params, grad_params = model_utils.combine_all_parameters(protos.enc, protos.dec, protos.classifier)
print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
  if name == 'enc' or name == 'dec' then
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
  end
end
-- encoder initial states
init_state = {}
for L=1,opt.num_layers do
  local h_init = torch.zeros(1, opt.rnn_size)
  if opt.gpuid >=0 then h_init = h_init:cuda() end
  table.insert(init_state, h_init:clone())
  table.insert(init_state, h_init:clone())
end

local init_state_global = clone_list(init_state)
--evaluation 
function eval_split(split_idx)
  print('evaluating loss over split index ' .. split_idx)
  local n = loader.split_sizes[split_idx]
  loader:reset_batch_pointer(split_idx)
  local correct_count = 0
  for i = 1,n do
    -- load data
    local x, y, label = loader:next_batch(1)
    if opt.gpuid >= 0 then
      x = x:float():cuda()
      y = y:float():cuda()
      label = label:float():cuda()
    end
    enc_length = x:size(2)
    dec_length = y:size(2)
    -- Forward pass
    -- 1) encoder
    local enc_out = {}
    local enc_state = {[0] = init_state_global}
    for t=1,enc_length do
      protos.enc:evaluate()
      local lst = protos.enc:forward({x[{{},t}], unpack(enc_state[t-1])})
      enc_state[t] = {}
      for i=1,#init_state do table.insert(enc_state[t], lst[i]) end
      table.insert(enc_out, lst[#lst])
    end
    -- 2) decoder
    local dec_out = {}
    local dec_state = {[0] = enc_state[enc_length]}
    for t=1,dec_length do
      protos.dec:evaluate()
      local lst = protos.dec:forward({y[{{},t}], unpack(dec_state[t-1])})
      dec_state[t] = {}
      for i=1,#init_state do table.insert(dec_state[t], lst[i]) end
      table.insert(dec_out, lst[#lst])
    end
    -- 3) classification
    local prediction = protos.classifier:forward({enc_out, dec_out})
    local max,indice = prediction:max(2)   -- indice is a 2d tensor here, we need to flatten it...
    if indice[1][1] == label[1] then correct_count = correct_count + 1 end
  end
  return correct_count*1.0/n
end

--training
function feval(x)
  if x ~= params then
    params:copy(x)
  end
  grad_params:zero()
  -- load data
  local x, y, label = loader:next_batch(1)
  if opt.gpuid >= 0 then
    x = x:float():cuda()
    y = y:float():cuda()
    label = label:float():cuda()
  end
  enc_length = x:size(2)
  dec_length = y:size(2) 
  -- Forward pass
  -- 1) encoder
  local enc_out = {}
  local enc_state = {[0] = init_state_global}
  for t=1,enc_length do
    clones.enc[t]:training()
    local lst = clones.enc[t]:forward({x[{{},t}], unpack(enc_state[t-1])})
    enc_state[t] = {}
    for i=1,#init_state do table.insert(enc_state[t], lst[i]) end
    table.insert(enc_out, lst[#lst])
  end
  -- 2) decoder
  local dec_out = {}
  local dec_state = {[0] = enc_state[enc_length]}
  for t=1,dec_length do
    clones.dec[t]:training()
    local lst = clones.dec[t]:forward({y[{{},t}], unpack(dec_state[t-1])})
    dec_state[t] = {}
    for i=1,#init_state do table.insert(dec_state[t], lst[i]) end
    table.insert(dec_out, lst[#lst])
  end
  -- 3) classification
  local prediction = protos.classifier:forward({enc_out, dec_out})
  local result = protos.criterion:forward(prediction, label)
  
  -- Backward pass
  -- 1) classification
  local dresult = protos.criterion:backward(prediction, label)
  local dprediction = protos.classifier:backward({enc_out, dec_out}, dresult)
  local denc_state = {[0]=clone_list(init_state, true)}
  local ddec_state = {[0]=clone_list(init_state, true)}
  for t=1,enc_length do
    denc_state[t] = clone_list(init_state, true)
    denc_state[t][#init_state] = dprediction[1][t]
  end 
  for t=1,dec_length do
    ddec_state[t] = clone_list(init_state, true)
    ddec_state[t][#init_state] = dprediction[2][t]
  end 
  -- 2) decoder
  for t=dec_length,1,-1 do
    local dlst = clones.dec[t]:backward({y[{{},t}], unpack(dec_state[t-1])}, ddec_state[t])
    for k,v in pairs(dlst) do
      if k > 1 then ddec_state[t-1][k-1]:add(v) end
    end
  end
  -- 3) encoder
  for i=1,#init_state do denc_state[enc_length][i]:add(ddec_state[0][i]) end
  for t=enc_length,1,-1 do
    local dlst = clones.enc[t]:backward({x[{{},t}], unpack(enc_state[t-1])}, denc_state[t])
    for k,v in pairs(dlst) do
      if k > 1 then denc_state[t-1][k-1]:add(v) end
    end
  end

  local grad_norm, shrink_factor
  grad_norm = torch.sqrt(grad_params:norm()^2)
  if grad_norm > opt.max_grad_norm then
    shrink_factor = opt.max_grad_norm / grad_norm
    grad_params:mul(shrink_factor)
  end
  params:add(grad_params:mul(-lr))
  return result 
end

-- start training
train_losses = {}
val_losses = {}
lr = opt.learningRate
local iterations = opt.max_epochs * loader.split_sizes[1]
for i = 1, iterations do
  -- train 
  local epoch = i / loader.split_sizes[1]
  local timer = torch.Timer()
  local time = timer:time().real
  train_losses[i] = feval(params)
  if i % opt.print_every == 0 then
    print(string.format("%d/%d (epoch %.2f), train_loss = %6.4f", i, iterations, epoch, train_losses[i]))
  end

  -- validate and save checkpoints
  if epoch == opt.max_epochs or i % opt.save_every == 0 then
    print ('evaluate on validation set')
    local val_loss = eval_split(2) -- 2 = validation
    print (val_loss)
    val_losses[#val_losses+1] = val_loss
    local savefile = string.format('%s/model_%s_epoch%.2f_%.2f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
    local checkpoint = {}
    checkpoint.protos = protos
    checkpoint.opt = opt
    checkpoint.train_losses = train_losses
    checkpoint.val_losses = val_losses
    checkpoint.vocab = {loader.idx2word, loader.word2idx}
    print('saving checkpoint to ' .. savefile)
    torch.save(savefile, checkpoint)
  end

  -- decay learning rate
  if i % loader.split_sizes[1] == 0 and #val_losses > 2 then
    if val_losses[#val_losses-1] - val_losses[#val_losses] < opt.decay_when then
      lr = lr * opt.decayRate
    end
  end

  -- misc
  if i%5==0 then collectgarbage() end
  if opt.time ~= 0 then
     print("Batch Time:", timer:time().real - time)
  end
end

-- end with test
test_loss = eval_split(3)
print (string.format("test_loss = %6.4f", test_loss))
