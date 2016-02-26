require 'torch'
require 'nn'
require 'nngraph'
require 'rnn'
model_utils = require 'util.model_utils'
BatchLoader = require 'util.BatchLoader'
require 'util.MaskedLoss'
require 'util.misc'
require 'util.CAveTable'
require 'optim'
require 'util.ReplicateAdd'
require 'util.LookupTableEmbedding_train'
classifier_simple = require 'model.classifier_simple'
encoder = require 'model.encoder_lstmn_w2v'
decoder = require 'model.decoder_deep_w2v'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-data_dir', 'data', 'path of the dataset')
cmd:option('-batch_size', '16', 'number of batches')
cmd:option('-max_epochs', 4, 'number of full passes through the training data')
cmd:option('-rnn_size', 400, 'dimensionality of sentence embeddings')
cmd:option('-word_vec_size', 300, 'dimensionality of word embeddings')
cmd:option('-dropout',0.4,'dropout. 0 = no dropout')
cmd:option('-seed',3435,'torch manual random number generator seed')
cmd:option('-max_length', 20, 'max length allowed for each sentence')
cmd:option('-print_every',1000,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every', 12500, 'save epoch')
cmd:option('-checkpoint_dir', 'cv4', 'output directory where checkpoints get written')
cmd:option('-savefile','model','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-checkpoint', 'checkpoint.t7', 'start from a checkpoint if a valid checkpoint.t7 file is given')
cmd:option('-learningRate', 0.001, 'learning rate')
cmd:option('-beta1', 0.9, 'momentum parameter 1')
cmd:option('-beta2', 0.999, 'momentum parameter 2')
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
loader = BatchLoader.create(opt.data_dir, opt.max_length, opt.batch_size)
opt.seq_length = loader.max_sentence_l 
opt.vocab_size = #loader.idx2word
opt.classes = 3
opt.word2vec = loader.word2vec
-- model
protos = {}
protos.enc = encoder.lstmn(opt.vocab_size, opt.rnn_size, opt.dropout, opt.word_vec_size, opt.batch_size, opt.word2vec)
protos.dec = decoder.lstmn(opt.vocab_size, opt.rnn_size, opt.dropout, opt.word_vec_size, opt.batch_size, opt.word2vec)
protos.criterion = nn.ClassNLLCriterion()
protos.classifier = classifier_simple.classifier(opt.rnn_size, opt.dropout, opt.classes)
-- ship to gpu
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end
-- params and grads
params, grad_params = model_utils.combine_all_parameters(protos.enc, protos.dec, protos.classifier)
print('number of parameters in the model: ' .. params:nElement())

function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'enc_lookup' then
      enc_lookup = layer
    elseif layer.name == 'dec_lookup' then
      dec_lookup = layer
    end
  end
end
protos.enc:apply(get_layer)
protos.dec:apply(get_layer)
--dec_lookup:share(enc_lookup, 'weight')

-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
  if name == 'enc' or name == 'dec' then
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.max_length, not proto.parameters)
  end
end
-- encoder/decoder initial states, decoder initial alignment vector
local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
if opt.gpuid >=0 then h_init = h_init:cuda() end


--evaluation 
function eval_split(split_idx)
  print('evaluating loss over split index ' .. split_idx)
  local n = loader.split_sizes[split_idx]
  loader:reset_batch_pointer(split_idx)
  local correct_count = 0
  for i = 1,n do
    -- load data
    local x, y, label = loader:next_batch(split_idx)
    if opt.gpuid >= 0 then
      x = x:float():cuda()
      y = y:float():cuda()
      label = label:float():cuda()
    end

    -- Forward pass
    -- 1) encoder
    local rnn_c_enc = {}
    local rnn_h_enc = {}
    table.insert(rnn_c_enc, h_init:clone())
    table.insert(rnn_h_enc, h_init:clone())
    for t=1,opt.max_length do
      clones.enc[t]:evaluate()
      local lst = clones.enc[t]:forward({x[{{},t}], narrow_list(rnn_c_enc, 1, t), narrow_list(rnn_h_enc, 1, t)})
      table.insert(rnn_c_enc, lst[1])
      table.insert(rnn_h_enc, lst[2])
    end
    -- 2) decoder
    local rnn_c_dec = {}
    local rnn_h_dec = {}
    local rnn_a = {[0] = h_init:clone()}
    local rnn_alpha = {[0] = h_init:clone()}
    table.insert(rnn_c_dec, rnn_c_enc[opt.max_length+1]:clone())
    table.insert(rnn_h_dec, rnn_h_enc[opt.max_length+1]:clone())
    for t=1,opt.max_length do
      clones.dec[t]:evaluate()
      local lst = clones.dec[t]:forward({y[{{},t}], rnn_a[t-1], rnn_alpha[t-1], narrow_list(rnn_c_dec, 1, t), narrow_list(rnn_h_dec, 1, t), rnn_c_enc, rnn_h_enc})
      table.insert(rnn_a, lst[1])
      table.insert(rnn_alpha, lst[2])
      table.insert(rnn_c_dec, lst[3])
      table.insert(rnn_h_dec, lst[4])
    end
    -- 3) classification
    protos.classifier:evaluate()
    local prediction = protos.classifier:forward({rnn_h_enc, rnn_h_dec})
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

  -- Forward pass
  -- 1) encoder
  local rnn_c_enc = {}
  local rnn_h_enc = {}
  table.insert(rnn_c_enc, h_init:clone())
  table.insert(rnn_h_enc, h_init:clone())
  for t=1,opt.max_length do
    clones.enc[t]:training()
    local lst = clones.enc[t]:forward({x[{{},t}], narrow_list(rnn_c_enc, 1, t), narrow_list(rnn_h_enc, 1, t)})
    table.insert(rnn_c_enc, lst[1])
    table.insert(rnn_h_enc, lst[2])
  end
  -- 2) decoder
  local rnn_c_dec = {}
  local rnn_h_dec = {}
  local rnn_a = {[0] = h_init:clone()}
  local rnn_alpha = {[0] = h_init:clone()}
  table.insert(rnn_c_dec, rnn_c_enc[opt.max_length+1]:clone())
  table.insert(rnn_h_dec, rnn_h_enc[opt.max_length+1]:clone())
  for t=1,opt.max_length do
    clones.dec[t]:training()
    local lst = clones.dec[t]:forward({y[{{},t}], rnn_a[t-1], rnn_alpha[t-1], narrow_list(rnn_c_dec, 1, t), narrow_list(rnn_h_dec, 1, t), rnn_c_enc, rnn_h_enc})
    table.insert(rnn_a, lst[1])
    table.insert(rnn_alpha, lst[2])
    table.insert(rnn_c_dec, lst[3])
    table.insert(rnn_h_dec, lst[4])
  end
  -- 3) classification
  protos.classifier:training()
  local prediction = protos.classifier:forward({rnn_h_enc, rnn_h_dec})
  local result = protos.criterion:forward(prediction, label)

  -- Backward pass
  -- 1) classification
  local dresult = protos.criterion:backward(prediction, label)
  local dprediction = protos.classifier:backward({rnn_h_enc, rnn_h_dec}, dresult)
  local drnn_alpha = clone_list(rnn_a, true) --true zeros
  local drnn_a = clone_list(rnn_a, true)
  local drnn_c_enc = clone_list(rnn_c_enc, true)
  local drnn_h_enc = clone_list(rnn_h_enc, true)
  local drnn_c_dec = clone_list(rnn_c_dec, true)
  local drnn_h_dec = clone_list(rnn_h_dec, true)

  for t=1,opt.max_length+1 do
      drnn_h_enc[t]:add(dprediction[1][t])
      drnn_h_dec[t]:add(dprediction[2][t])
  end

  -- 2) decoder
  for t=opt.max_length,1,-1 do
    local dlst = clones.dec[t]:backward({y[{{},t}], rnn_a[t-1], rnn_alpha[t-1], narrow_list(rnn_c_dec, 1, t), narrow_list(rnn_h_dec, 1, t), rnn_c_enc, rnn_h_enc}, {drnn_a[t], drnn_alpha[t], drnn_c_dec[t+1], drnn_h_dec[t+1]})
    drnn_a[t-1]:add(dlst[2])
    drnn_alpha[t-1]:add(dlst[3])
    for k=1, t do
      drnn_c_dec[k]:add(dlst[4][k])    
      drnn_h_dec[k]:add(dlst[5][k])    
    end
    for k=1, opt.max_length+1 do
      drnn_c_enc[k]:add(dlst[6][k])
      drnn_h_enc[k]:add(dlst[7][k])
    end
  end

  -- 3) encoder
  drnn_c_enc[opt.max_length+1]:add(drnn_c_dec[1])
  drnn_h_enc[opt.max_length+1]:add(drnn_h_dec[1])
  for t=opt.max_length,1,-1 do
    dlst = clones.enc[t]:backward({x[{{},t}], narrow_list(rnn_c_enc, 1, t), narrow_list(rnn_h_enc, 1, t)}, {drnn_c_enc[t+1], drnn_h_enc[t+1]})
    for k=1, t do
      drnn_c_enc[k]:add(dlst[2][k])    
      drnn_h_enc[k]:add(dlst[3][k])    
    end
  end

  local grad_norm, shrink_factor
  grad_norm = torch.sqrt(grad_params:norm()^2)
  if grad_norm > opt.max_grad_norm then
    shrink_factor = opt.max_grad_norm / grad_norm
    grad_params:mul(shrink_factor)
  end
  return result, grad_params 
end

-- start training
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learningRate, beta1 = opt.beta1, beta2 = opt.beta2}
local iterations = opt.max_epochs * loader.split_sizes[1]
for i = 1, iterations do
  -- train 
  local epoch = i / loader.split_sizes[1]
  local timer = torch.Timer()
  local time = timer:time().real
  local _, loss = optim.adam(feval, params, optim_state)
  train_losses[i] = loss[1]
  if i % opt.print_every == 0 then
    print(string.format("%d/%d (epoch %.2f), train_loss = %6.4f", i, iterations, epoch, train_losses[i]))
  end

  -- validate and save checkpoints
  if epoch == opt.max_epochs or i % opt.save_every == 0 then
    print ('evaluate on validation set')
    local val_loss = eval_split(2) -- 2 = validation
    print (val_loss)
    if epoch>1.5 then
      local test_loss = eval_split(3) -- 3 = test
      print (test_loss)
    end
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
      opt.learningRate = opt.learningRate * opt.decayRate
    end
  end

  -- index 1 is zero
  enc_lookup.weight[1]:zero()
  enc_lookup.gradWeight[1]:zero()
  dec_lookup.weight[1]:zero()
  dec_lookup.gradWeight[1]:zero()

  -- misc
  if i%5==0 then collectgarbage() end
  if opt.time ~= 0 then
     print("Batch Time:", timer:time().real - time)
  end
end

-- end with test
test_loss = eval_split(3)
print (string.format("test_loss = %6.4f", test_loss))
