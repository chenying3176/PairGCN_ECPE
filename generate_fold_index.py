import numpy as np

np.random.seed(1234)

all_train_index = []
all_dev_index = []
all_test_index = []
count = 0
with open('data/splited_data/fold_index.txt', 'r') as f:
	for line in f:
		line = line.strip()
		if line == '':
			continue
		if count % 2 == 0:
			index = line.split()[1:]
			np.random.shuffle(index)
			size = len(index) // 9
			dev_index, train_index = index[:size], index[size:]
			all_train_index.append(['train_%d'%(count//2+1)]+sorted([int(z) for z in train_index]))
			all_dev_index.append(['dev_%d'%(count//2+1)]+sorted([int(z) for z in dev_index]))
		else:
			all_test_index.append(line)
		count += 1

with open('data/splited_data/new_fold_index.txt', 'w', encoding='utf-8') as f:
	for i in range(len(all_train_index)):
		f.write(' '.join([str(k) for k in all_train_index[i]])+'\n')
		f.write(' '.join([str(k) for k in all_dev_index[i]])+'\n')
		f.write(all_test_index[i]+'\n')
