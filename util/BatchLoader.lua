local BatchLoader = {}
local stringx = require('pl.stringx')
BatchLoader.__index = BatchLoader


function BatchLoader.labelToNumber(label)
   local result
   if label == '-' then
      result = 3
      print ('labelToNum sees label -', label)
      -- returning 3 will cause program to abort (it is expecting values 0..2)
      return result
   end
   if label == 'neutral' then
      result = 0
      return result
   end
   if label == 'contradiction' then
      result = 1
      return result
   end
   if label == 'entailment' then
      result = 2
      return result
   end
   print ('labelToNum sees unknown label', label)
   -- returning -1 will cause program to abort.
   return -1
end

function BatchLoader.create(data_dir, max_sentence_l , batch_size)
    local self = {}
    setmetatable(self, BatchLoader)
    local train_file = path.join(data_dir, 'train.txt')
    local valid_file = path.join(data_dir, 'dev.txt')
    local test_file = path.join(data_dir, 'test.txt')
    local input_files = {train_file, valid_file, test_file}
    local input_w2v = path.join(data_dir, 'word2vec.txt')
    

    -- construct a tensor with all the data
    local s1, s2, label, idx2word, word2idx, word2vec = BatchLoader.text_to_tensor(input_files, max_sentence_l, input_w2v)
    self.max_sentence_l = max_sentence_l
    self.idx2word, self.word2idx = idx2word, word2idx
    self.vocab_size = #self.idx2word 
    self.word2vec = word2vec
    self.split_sizes = {}
    self.all_batches = {}
 
    print(string.format('Word vocab size: %d', #self.idx2word))
    -- cut off the end for train/valid sets so that it divides evenly
    for split=1,3 do
       local s1data = s1[split]
       local s2data = s2[split]
       local label_data = label[split]
       local len = s1data:size(1)
       if len % (batch_size) ~= 0 then
          s1data = s1data:sub(1, batch_size * math.floor(len / batch_size))
          s2data = s2data:sub(1, batch_size * math.floor(len / batch_size))
          label_data = label_data:sub(1, batch_size * math.floor(len / batch_size))   --let's just make the batch_size a multiplier of the test data size
       end

       s1_batches = s1data:split(batch_size,1)
       s2_batches = s2data:split(batch_size,1)
       label_batches = label_data:split(batch_size,1)
       nbatches = #s1_batches
       self.split_sizes[split] = nbatches
       self.all_batches[split] = {s1_batches, s2_batches, label_batches}
    end
 
    self.batch_idx = {0,0,0}
    print(string.format('data load done. Number of batches in train: %d, val: %d, test: %d', self.split_sizes[1], self.split_sizes[2], self.split_sizes[3]))
    collectgarbage()
    return self
end

function BatchLoader:reset_batch_pointer(split_idx, batch_idx)
    batch_idx = batch_idx or 0
    self.batch_idx[split_idx] = batch_idx
end

function BatchLoader:next_batch(split_idx)
    -- split_idx is integer: 1 = train, 2 = val, 3 = test
    self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
    if self.batch_idx[split_idx] > self.split_sizes[split_idx] then
        self.batch_idx[split_idx] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local idx = self.batch_idx[split_idx]
    return self.all_batches[split_idx][1][idx], self.all_batches[split_idx][2][idx], self.all_batches[split_idx][3][idx]
end

function BatchLoader.text_to_tensor(input_files, max_sentence_l, input_w2v)
    print('Processing text into tensors...')
    local f
    local vocab_count = {} -- vocab count 
    local idx2word = {'ZERO', 'START'} 
    local word2idx = {}; word2idx['ZERO'] = 1; word2idx['START'] = 2
    local split_counts = {}
    local output_tensors1 = {}  --for sentence1
    local output_tensors2 = {}  -- for sentence2
    local labels = {}
    -- first go through train/valid/test to get max sentence length
    -- also counts the number of sentences
    for	split = 1,3 do -- split = 1 (train), 2 (val), or 3 (test)
       f = io.open(input_files[split], 'r')       
       local scounts = 0
       for line in f:lines() do
          scounts = scounts + 1
       end
       f:close()
       split_counts[split] = scounts  --the number of sentences in each split
    end
      
    print(string.format('Token count: train %d, val %d, test %d', 
    			split_counts[1], split_counts[2], split_counts[3]))
    
    for	split = 1,3 do -- split = 1 (train), 2 (val), or 3 (test)     
       -- Preallocate the tensors we will need.
       -- Watch out the second one needs a lot of RAM.
       output_tensors1[split] = torch.ones(split_counts[split], max_sentence_l):long() 
       output_tensors2[split] = torch.ones(split_counts[split], max_sentence_l):long() 
       labels[split] = torch.zeros(split_counts[split]):long() 
       -- process each file in split
       f = io.open(input_files[split], 'r')
       local sentence_num = 0
       for line in f:lines() do
          sentence_num = sentence_num + 1
          local triplet = stringx.split(line, '\t')
          local label, s1, s2 = triplet[1], triplet[2], triplet[3]
          labels[split][sentence_num] = BatchLoader.labelToNumber(label) + 1
          -- append tokens in the sentence1
          output_tensors1[split][sentence_num][1] = 2
          local word_num = 1
          for rword in s1:gmatch'([^%s]+)' do
             word_num = word_num + 1
             if word2idx[rword]==nil then
                idx2word[#idx2word + 1] = rword 
                word2idx[rword] = #idx2word
             end
             output_tensors1[split][sentence_num][word_num] = word2idx[rword]
             if word_num == max_sentence_l then break end
          end
          -- append tokens in the sentence2
          output_tensors2[split][sentence_num][1] = 2
          word_num = 1
          for rword in s2:gmatch'([^%s]+)' do
             word_num = word_num + 1
             if word2idx[rword]==nil then
                idx2word[#idx2word + 1] = rword 
                word2idx[rword] = #idx2word
             end
             output_tensors2[split][sentence_num][word_num] = word2idx[rword]
             if word_num == max_sentence_l then break end
          end
       end
    end

    local w2v = {}
    local w2v_file = io.open(input_w2v, 'r')
    for line in w2v_file:lines() do
        tokens = stringx.split(line, ' ')
        word = tokens[1]
        if word2idx[word] ~= nil then
            w2v[word2idx[word]] = torch.zeros(300) 
            for tid=2,301 do
                w2v[word2idx[word]][tid-1] = tonumber(tokens[tid])
            end
        end
    end
    w2v_file:close()

    return output_tensors1, output_tensors2, labels, idx2word, word2idx, w2v
end

return BatchLoader

