local decoder_deep_w2v = {}
local ok, cunn = pcall(require, 'fbcunn')
if nn.LookupTableEmbedding_train then
  LookupTable = nn.LookupTableEmbedding_train
elseif nn.LookupTableEmbedding_fixed then
  LookupTable = nn.LookupTableEmbedding_fixed
else
  LookupTable = nn.LookupTableEmbedding_update
end

function decoder_deep_w2v.lstmn(input_size, rnn_size, dropout, word_emb_size, batch_size, word2vec)
  -- input_size : vocab size
  dropout = dropout or 0
  local vec_size = word_emb_size or rnn_size
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- prev intra allignment
  table.insert(inputs, nn.Identity()()) -- prev inter allignment
  table.insert(inputs, nn.Identity()()) -- prev_c_table [c0, c1, c2... c(t-1)] for decoder
  table.insert(inputs, nn.Identity()()) -- prev_h_table [h0, h1, h2... h(t-1)] for decoder
  table.insert(inputs, nn.Identity()()) -- enc_c_table [c0, c1, c2... cm] for encoder
  table.insert(inputs, nn.Identity()()) -- enc_h_table [h0, h1, h2... hm] for encoder

  -- inputs
  local x, word_vec
  local outputs = {}
  local prev_a_intra = inputs[2]
  local prev_a_inter = inputs[3]
  local prev_c_table = inputs[4]
  local prev_h_table = inputs[5]
  local enc_c_table = inputs[6]
  local enc_h_table = inputs[7]
  word_vec_layer = LookupTable(input_size, vec_size, word2vec)
  word_vec_layer.name = 'dec_lookup'
  word_vec = word_vec_layer(inputs[1])

  x = nn.Identity()(word_vec)
  local x_all = nn.Linear(vec_size, 7 * rnn_size)(x)

  -- intra attention
  local prev_h_join = nn.JoinTable(2)(prev_h_table)
  local prev_c_join = nn.JoinTable(2)(prev_c_table)
  local intra_x = nn.Narrow(2, 1, rnn_size)(x_all)
  local intra_a = nn.Linear(rnn_size, rnn_size)(prev_a_intra)
  intra_x = nn.CAddTable()({intra_x, intra_a})
  local intra_h = nn.Linear(rnn_size, rnn_size)(nn.View(-1, rnn_size)(prev_h_join))   
  intra_h = nn.View(batch_size, -1)(intra_h)
  local intra_sum = nn.Tanh()(nn.AddScalar()({intra_h, intra_x}))
  intra_sum = nn.View(-1, rnn_size)(intra_sum)
  local intra_score = nn.Linear(rnn_size, 1)(intra_sum)  
  intra_score = nn.View(batch_size, -1)(intra_score)
  intra_score = nn.SoftMax(2)(intra_score) 
  intra_score = nn.View(batch_size, 1, -1)(intra_score)
  prev_h_join = nn.View(batch_size, -1, rnn_size)(prev_h_join)
  prev_c_join = nn.View(batch_size, -1, rnn_size)(prev_c_join)
  local prev_h = nn.View(batch_size, rnn_size)(nn.MM(false, false)({intra_score, prev_h_join}))  --this is the allignment vector at time step t
  local prev_c = nn.View(batch_size, rnn_size)(nn.MM(false, false)({intra_score, prev_c_join}))

  -- inter attention
  local enc_h_join = nn.JoinTable(2)(enc_h_table)
  local enc_c_join = nn.JoinTable(2)(enc_c_table)
  local inter_x = nn.Narrow(2, rnn_size + 1, rnn_size)(x_all)
  local inter_a = nn.Linear(rnn_size, rnn_size)(prev_a_inter)
  inter_x = nn.CAddTable()({inter_x, inter_a})
  local inter_h = nn.Linear(rnn_size, rnn_size)(nn.View(-1, rnn_size)(enc_h_join))   
  inter_h = nn.View(batch_size, -1)(inter_h)
  local inter_sum = nn.Tanh()(nn.AddScalar()({inter_h, inter_x}))
  inter_sum = nn.View(-1, rnn_size)(inter_sum)
  local inter_score = nn.Linear(rnn_size, 1)(inter_sum)  
  inter_score = nn.View(batch_size, -1)(inter_score)
  inter_score = nn.SoftMax(2)(inter_score) 
  inter_score = nn.View(batch_size, 1, -1)(inter_score)
  enc_h_join = nn.View(batch_size, -1, rnn_size)(enc_h_join)
  enc_c_join = nn.View(batch_size, -1, rnn_size)(enc_c_join)
  local prev_enc = nn.View(batch_size, rnn_size)(nn.MM(false, false)({inter_score, enc_h_join}))  --this is the allignment vector at time step t
  local prev_mem = nn.View(batch_size, rnn_size)(nn.MM(false, false)({inter_score, enc_c_join}))  --this is the allignment vector at time step t
  
  -- LSTM misc
  local i2h = nn.Narrow(2, 2 * rnn_size + 1, 4 * rnn_size)(x_all)
  local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
  local all_input_sums = nn.CAddTable()({i2h, h2h})
  local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums)
  sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
  local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
  local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
  local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)
  local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums)
  local x2h =  nn.Narrow(2, 6 * rnn_size + 1, rnn_size)(x_all)
  local e2h = nn.Linear(rnn_size, rnn_size)(prev_enc)
  local cast_gate = nn.Sigmoid()(nn.CAddTable()({x2h, e2h}))
  in_transform = nn.Tanh()(in_transform)
  local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform}),
        nn.CMulTable()({cast_gate,   prev_mem}),
    })
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  
  -- outputs
  table.insert(outputs, prev_h)  -- intra a
  table.insert(outputs, prev_enc) -- inter a
  table.insert(outputs, next_c)
  table.insert(outputs, next_h)

  return nn.gModule(inputs, outputs)
end

return decoder_deep_w2v
