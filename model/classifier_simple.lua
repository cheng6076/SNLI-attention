local classifier_simple = {}

function classifier_simple.classifier(rnn_size, dropout, classes)
  local dropout = dropout or 0 
  local rnn_size = rnn_size
  local classes = classes
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- h_enc
  table.insert(inputs, nn.Identity()()) -- h_dec
  local h_enc_ave = nn.CAveTable()(inputs[1])
  local h_dec_ave = nn.CAveTable()(inputs[2])
  local h_concat = nn.JoinTable(2)({h_enc_ave, h_dec_ave})
  local top_h = nn.ReLU()(nn.Linear(2*rnn_size, rnn_size)(h_concat))
  top_h = nn.ReLU()(nn.Linear(rnn_size, rnn_size)(top_h))
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, classes)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  return nn.gModule(inputs, {logsoft})
end

return classifier_simple
