local classifier_simple = {}

function classifier_simple.classifier(rnn_size, dropout, classes)
  local dropout = dropout or 0 
  local rnn_size = rnn_size
  local classes = classes
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- h_enc
  table.insert(inputs, nn.Identity()()) -- h_dec
  table.insert(inputs, nn.Identity()()) -- mx
  table.insert(inputs, nn.Identity()()) -- my
  local h_enc_sum = nn.CAddTable(1, 1)(inputs[1])
  local h_dec_sum = nn.CAddTable(1, 1)(inputs[2])
  local h_enc_cnt = nn.Replicate(rnn_size, 2, 1)(nn.Sum(1, 1)(inputs[3]))
  local h_dec_cnt = nn.Replicate(rnn_size, 2, 1)(nn.Sum(1, 1)(inputs[4]))
  local h_enc_ave = nn.CDivTable(1, 1)({h_enc_sum, h_enc_cnt})
  local h_dec_ave = nn.CDivTable(1, 1)({h_dec_sum, h_dec_cnt})
  local h_concat = nn.JoinTable(1, 1)({h_enc_ave, h_dec_ave})
  local top_h = nn.ReLU()(nn.Linear(3*rnn_size, rnn_size)(h_concat))
  top_h = nn.ReLU()(nn.Linear(rnn_size, rnn_size)(top_h))
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, classes)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  return nn.gModule(inputs, {logsoft})
end

return classifier_simple
