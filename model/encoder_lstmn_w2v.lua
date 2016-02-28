local encoder_lstmn_w2v = {}
if nn.LookupTableEmbedding_train then
  LookupTable = nn.LookupTableEmbedding_train
elseif nn.LookupTableEmbedding_fixed then
  LookupTable = nn.LookupTableEmbedding_fixed
else
  LookupTable = nn.LookupTableEmbedding_update
end

function encoder_lstmn_w2v.lstmn(input_size, rnn_size, dropout, word_emb_size, batch_size, word2vec)
  -- input_size : vocab size
  dropout = dropout or 0
  local vec_size = word_emb_size or rnn_size
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- prev_c_table [c0, c1, c2... c(t-1)]
  table.insert(inputs, nn.Identity()()) -- prev_h_table [h0, h1, h2... h(t-1)]

  local x, input_size_L, word_vec
  local outputs = {}

  -- c,h from previous timesteps
  local prev_c_table = inputs[2]
  local prev_h_table = inputs[3]

  -- the input to this layer
  word_vec_layer = LookupTable(input_size, vec_size, word2vec)
  word_vec_layer.name = 'enc_lookup'
  word_vec = word_vec_layer(inputs[1])
  x = nn.Identity()(word_vec)
  input_size_L = vec_size

  local prev_h_join = nn.JoinTable(2)(prev_h_table)
  local prev_c_join = nn.JoinTable(2)(prev_c_table)
  local attention_x = nn.Linear(input_size_L, rnn_size)(x)
  local attention_h = nn.Linear(rnn_size, rnn_size)(nn.View(-1, rnn_size)(prev_h_join))   
  attention_h = nn.View(batch_size, -1)(attention_h)
  local attention_sum = nn.Tanh()(nn.AddScalar()({attention_h, attention_x}))
  attention_sum = nn.View(-1, rnn_size)(attention_sum)
  local attention_score = nn.Linear(rnn_size, 1)(attention_sum)  
  attention_score = nn.View(batch_size, -1)(attention_score)
  attention_score = nn.SoftMax(2)(attention_score) 
  attention_score = nn.View(batch_size, 1, -1)(attention_score)

  prev_h_join = nn.View(batch_size, -1, rnn_size)(prev_h_join)
  prev_c_join = nn.View(batch_size, -1, rnn_size)(prev_c_join)
  local prev_h = nn.View(batch_size, rnn_size)(nn.MM(false, false)({attention_score, prev_h_join}))  --this is the allignment vector at time step t
  local prev_c = nn.View(batch_size, rnn_size)(nn.MM(false, false)({attention_score, prev_c_join}))

  -- evaluate the input sums at once for efficiency
  local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x)
  local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
  local all_input_sums = nn.CAddTable()({i2h, h2h})
    -- decode the gates
  local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums)
  sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
  local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
  local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
  local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)
    -- decode the write inputs
  local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums)
  in_transform = nn.Tanh()(in_transform)

    -- perform the LSTM update
  local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    -- gated cells form the output
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
  table.insert(outputs, next_c)
  table.insert(outputs, next_h)

  return nn.gModule(inputs, outputs)
end

return encoder_lstmn_w2v
