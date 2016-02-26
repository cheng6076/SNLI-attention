import numpy as np
w1 = {}
vec = open('glove.840B.300d.txt', 'r')
for line in vec.readlines():
  line=line.split(' ')
  w1[line[0]] = np.asarray([float(x) for x in line[1:]])
vec.close()
w2 = {}
f1 = open('/disk/scratch/s1537177/entailment/data/train.txt','r')
f2 = open('/disk/scratch/s1537177/entailment/data/dev.txt','r')
f3 = open('/disk/scratch/s1537177/entailment/data/test.txt','r')
f = [f1, f2, f3]

for file in f:
  for line in file.readlines():
    line=line.split('\t')
    s1 = line[1].rstrip().split(' ')
    s2 = line[2].rstrip().split(' ')
    for word in s1:
      if not w1.has_key(word):
        if not w2.has_key(word):w2[word]=[]
        for neighbor in s1:
          if w1.has_key(neighbor):
            w2[word].append(w1[neighbor])
    for word in s2:
      if not w1.has_key(word):
        if not w2.has_key(word):w2[word]=[]
        for neighbor in s2:
          if w1.has_key(neighbor):
            w2[word].append(w1[neighbor])
  file.close()

for k in w2.iterkeys():
  w2[k] = list(sum(w2[k])/len(w2[k]))

s = open('oovvec', 'w')
for k in w2.iterkeys():
  s.write(k+' '+' '.join([str(x) for x in w2[k]])+'\n')
s.close()


