--standard LSTM with fixed embeddings
local BatchLoaderB = {}
local stringx = require('pl.stringx')
BatchLoaderB.__index = BatchLoaderB

function BatchLoaderB.create(data_dir)
    local self = {}
    setmetatable(self, BatchLoader)
    local train_file = path.join(data_dir, 'train.txt')
    local valid_file = path.join(data_dir, 'dev.txt')
    local test_file = path.join(data_dir, 'test.txt')
    local input_files = {train_file, valid_file, test_file}
    local input_w2v = path.join(data_dir, 'word2vec.txt')
    -- construct a tensor with all the data
    local s1, s2, label, max_sentence_l, idx2word, word2idx, word2vec = BatchLoader.text_to_tensor(input_files, input_w2v)
    self.max_sentence_l = max_sentence_l
    self.idx2word, self.word2idx = idx2word, word2idx
    self.vocab_size = #self.idx2word 
    self.word2vec = word2vec 
    print(string.format('Word vocab size: %d', #self.idx2word))
    print(string.format('Word vec coverage: %d', #self.word2vec))
    -- cut off the end for train/valid sets so that it divides evenly
    self.all_batches = {s1, s2, label}
    self.split_sizes = {#s1[1], #s1[2], #s1[3]} 
    self.batch_idx = {0,0,0}
    print(string.format('data load done. Number of batches in train: %d, val: %d, test: %d', self.split_sizes[1], self.split_sizes[2], self.split_sizes[3]))
    collectgarbage()
    return self
end

function BatchLoaderB:reset_batch_pointer(split_idx, batch_idx)
    batch_idx = batch_idx or 0
    self.batch_idx[split_idx] = batch_idx
end

function BatchLoaderB:next_batch(split_idx)
    -- split_idx is integer: 1 = train, 2 = val, 3 = test
    self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
    if self.batch_idx[split_idx] > self.split_sizes[split_idx] then
        self.batch_idx[split_idx] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local idx = self.batch_idx[split_idx]
    return self.all_batches[1][split_idx][idx], self.all_batches[2][split_idx][idx], self.all_batches[3][split_idx][idx]
end

function BatchLoaderB.text_to_tensor(input_files, input_w2v)
    print('Processing text into tensors...')
    local f
    local vocab_count = {} -- vocab count 
    local max_sentence_l = 0 -- max sentence length
    local idx2word = {} 
    local word2idx = {}; 
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
          if split==1 then  --we don't care the max sentence length is dev or test
            local wcounts = 0
            local triplet = stringx.split(line, '\t')
            local label, s1, s2 = triplet[1], triplet[2], triplet[3]
            for word in s1:gmatch'([^%s]+)' do
	       wcounts = wcounts + 1
            end
            max_sentence_l = math.max(max_sentence_l, wcounts)
            -- we find the longest sentence in general, just to save some efforts
            wcount = 0
            for word in s2:gmatch'([^%s]+)' do
	       wcounts = wcounts + 1
            end
            max_sentence_l = math.max(max_sentence_l, wcounts)
          end
       end
       f:close()
       split_counts[split] = scounts  --the number of sentences in each split
    end
      
    print('After first pass of data, max sentence length is: ' .. max_sentence_l)
    print(string.format('Token count: train %d, val %d, test %d', 
    			split_counts[1], split_counts[2], split_counts[3]))
    
    for	split = 1,3 do -- split = 1 (train), 2 (val), or 3 (test)     
       -- Preallocate the tensors we will need.
       -- Watch out the second one needs a lot of RAM.
       output_tensors1[split] = {} 
       output_tensors2[split] = {} 
       labels[split] = {} 
       -- process each file in split
       f = io.open(input_files[split], 'r')
       local sentence_num = 0
       for line in f:lines() do
          sentence_num = sentence_num + 1
          local triplet = stringx.split(line, '\t')
          local label, s1, s2 = triplet[1], triplet[2], triplet[3]
          label = torch.Tensor({tonumber(label) + 1})  --0 is not allowed in torch
          labels[split][sentence_num] = label
          -- count word and create zero tensor for sentence1
          local word_count = 0
          for rword in s1:gmatch'([^%s]+)' do
            word_count = word_count + 1
          end
          output_tensors1[split][sentence_num] = torch.zeros(1, word_count)  --make it 2d just to be compatible with some nn modules
          -- append tokens in the sentence1
          local word_num = 0
          for rword in s1:gmatch'([^%s]+)' do
             word_num = word_num + 1
             if word2idx[rword]==nil then
                idx2word[#idx2word + 1] = rword 
                word2idx[rword] = #idx2word
             end
             output_tensors1[split][sentence_num][1][word_num] = word2idx[rword]
          end
          -- count word and create zero tensor for sentence2
          word_count = 0
          for rword in s2:gmatch'([^%s]+)' do
            word_count = word_count + 1
          end
          output_tensors2[split][sentence_num] = torch.zeros(1, word_count)
          -- append tokens in the sentence1
          word_num = 0
          for rword in s2:gmatch'([^%s]+)' do
             word_num = word_num + 1
             if word2idx[rword]==nil then
                idx2word[#idx2word + 1] = rword 
                word2idx[rword] = #idx2word
             end
             output_tensors2[split][sentence_num][1][word_num] = word2idx[rword]
          end
       end
       f:close()
    end

    local w2v = {}
    local w2v_file = io.open(input_w2v, 'r')
    for line in w2v_file:lines() do
        tokens = stringx.split(line, ' ')
        word = tokens[1]
        if word2idx[word] ~= nil then
            w2v[word2idx[word]] = torch.zeros(300)  --fixed for google news vectors
            for tid=2,301 do
                w2v[word2idx[word]][tid-1] = tonumber(tokens[tid])
            end
        end
    end
    w2v_file:close()

    return output_tensors1, output_tensors2, labels, max_sentence_l, idx2word, word2idx, w2v
end

return BatchLoaderB

