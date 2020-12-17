import os
import torch
import random
import json
import numpy as np
import pickle
import linecache
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

float_tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
long_tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

def generate_adj_matrix(slens, max_slen, window_size=1):
	matrix = np.zeros((len(slens), 3, max_slen, max_slen, max_slen))
	for cur_idx, sl in enumerate(slens):
		for i in range(sl):
			start = max(0, i-2)
			end = min(i+3, sl)
			for j in range(start, end):
				for l in range(max(j-2, start), min(j+3, end)):
					diff = abs(j - l)
					if diff == 0:
						continue
					matrix[cur_idx, 0, i, j, l] = 1. / diff
	return matrix

class CausePairLoader():
	def __init__(self, inp_files, device):
		self.raw_data = CausePairRawData(inp_files)
		self.device = device
		self.max_sen_len = inp_files['max_sen_len']
		self.tmp_dir = inp_files['tmp_dir']
		if not os.path.exists(self.tmp_dir):
			os.mkdir(self.tmp_dir)
		self.word_list, self.emotion, self.max_clause_num = self.load_vocab(self.raw_data.data, tmp_dir=self.tmp_dir)
		self.emb = self.load_pretrained_embedding(inp_files['emb_dir'], inp_files['vocab_dir'], self.word_list, tmp_dir=self.tmp_dir)
		self.word2id = {word:idx for idx, word in enumerate(self.word_list)}
		self.id2word = {idx:word for idx, word in enumerate(self.word_list)}
		self.emo2id = {emo:idx for idx, emo in enumerate(self.emotion)}

	def generate_loader(self, data, batch_size, shuffle, drop_last, mode='train'):
		def generate_batch(data, sample_size, batch_size):
			return [data[i:i+batch_size] for i in range(0, sample_size, batch_size)]
		ids, clens, slens, labels, emotions, events = data
		sample_size = len(ids)
		
		max_clens = [max(cl) for cl in clens]
		sort_idx = np.argsort(max_clens)
		ids = [ids[index] for index in sort_idx]
		clens = [clens[index] for index in sort_idx]
		slens = [slens[index] for index in sort_idx]
		labels = [labels[index] for index in sort_idx]
		emotions = [emotions[index] for index in sort_idx]
		events = [events[index] for index in sort_idx]

		ids_batch = generate_batch(ids, sample_size, batch_size)
		clens_batch = generate_batch(clens, sample_size, batch_size)
		slens_batch = generate_batch(slens, sample_size, batch_size)
		labels_batch = generate_batch(labels, sample_size, batch_size)
		emotions_batch = generate_batch(emotions, sample_size, batch_size)
		events_batch = generate_batch(events, sample_size, batch_size)

		assert(len(ids_batch) == \
			len(labels_batch) == \
			len(clens_batch) == \
			len(slens_batch) == \
			len(emotions_batch) == \
			len(events_batch))

		all_batch = (ids_batch, clens_batch, slens_batch, labels_batch, emotions_batch, events_batch)

		tensor_batch = self._get_tensor(all_batch, mode)
		return tensor_batch

	def generate_fold_data(self, fold_num, batch_size, shuffle, drop_last):
		tmp_data_dir = os.path.join(self.tmp_dir, 'tmp_data_%d'%fold_num)

		if not os.path.exists(tmp_data_dir) or shuffle:
			cur_fold_index = self.raw_data.fold_index[fold_num]
			train_index, dev_index, test_index = cur_fold_index

			train_data, dev_data, test_data = \
			[self.raw_data.data[index] for index in train_index if index in self.raw_data.data], \
			[self.raw_data.data[index] for index in dev_index if index in self.raw_data.data], \
			[self.raw_data.data[index] for index in test_index if index in self.raw_data.data]

			train_data = self._convert_sample2id(train_data)
			dev_data = self._convert_sample2id(dev_data)
			test_data = self._convert_sample2id(test_data)

			train_loader = self.generate_loader(train_data, batch_size, shuffle, drop_last)
			dev_loader = self.generate_loader(dev_data, batch_size, shuffle, drop_last, mode='dev')
			test_loader = self.generate_loader(test_data, batch_size, shuffle, drop_last, mode='test')
		else:
			train_loader = torch.load(tmp_data_dir+'.train')
			test_loader = torch.load(tmp_data_dir+'.test')
		return train_loader, dev_loader, test_loader

	def _get_tensor(self, batch, mode='train'):
		ids_batch, clens_batch, \
		slens_batch, labels_batch, emotions_batch, events_batch = batch
		
		ids_tensor_batch = []
		clen_tensor_batch = []
		slen_tensor_batch = []
		sgraph_tensor_batch = []
		label_tensor_batch = []
		emotion_tensor_batch = []
		event_tensor_batch = []
		distance_tensor_batch = []

		for ids, clen, slen, labels, emotions, events in zip(*batch):
			cids_batch = []
			clen_batch = []
			label_batch = []
			emotion_batch = []
			event_batch = []
			distance_batch = []

			max_clen = max([max(x) for x in clen])
			max_slen = max(slen)
			adj_matrix = generate_adj_matrix(slen, max_slen)
			for cur_idx, cids, cl, sl, label, emotion, event \
				in zip(range(len(slen)), ids, clen, slen, labels, emotions, events):
				cids_batch.extend([cids[i]+[0]*(max_clen-cl[i]) for i in range(len(cids))])
				clen_batch.extend(cl)
				# if mode != 'train':
				# 	label_mat = np.zeros((max_slen, max_slen))
				# 	for l in label:
				# 		start, end = l.replace('(', '').replace(')', '').split(',')
				# 		label_mat[int(start)-1, int(end)-1] = 1
				# 	label_mat[sl:, :] = -1
				# 	label_mat[:, sl:] = -1
				# else:
				# 	label_mat = -np.ones((max_slen, max_slen))
				# 	for zz in range(sl):
				# 		label_mat[zz, max(0, zz-2):min(sl, zz+3)] = 0
				# 	for l in label:
				# 		start, end = l.replace('(', '').replace(')', '').split(',')
				# 		label_mat[int(start)-1, int(end)-1] = 1
				label_mat = np.zeros((max_slen, max_slen))
				for l in label:
					start, end = l.replace('(', '').replace(')', '').split(',')
					label_mat[int(start)-1, int(end)-1] = 1
				label_mat[sl:, :] = -1
				label_mat[:, sl:] = -1
				label_batch.append(label_mat)

				distance_mat = np.arange(max_slen).reshape(max_slen, 1)
				distance_mat = distance_mat.repeat(max_slen, axis=1)
				position = np.arange(max_slen)
				distance_mat = np.minimum(np.abs(distance_mat - position) + 1, 3)
				distance_mat[sl:, :] = 0
				distance_mat[:, sl:] = 0				
				distance_batch.append(distance_mat)

				semo_batch = []
				for emo in emotion:
					cemo = [0 for _ in range(len(self.emo2id))]
					if emo != 'null':
						emo = emo.split('&')
						for ce in emo:
							cemo[self.emo2id[ce]] = 1
					emotion_batch.append(cemo)

				event_batch.extend(event)

			cids_tensor = long_tensor(cids_batch, device=self.device)
			sgraph_tensor = float_tensor(adj_matrix, device=self.device).requires_grad_(False)
			clen_tensor = long_tensor(clen_batch, device=self.device).requires_grad_(False)
			slen_tensor = long_tensor(slen, device=self.device).requires_grad_(False)
			distance_tensor = long_tensor(distance_batch, device=self.device)
			label_tensor = float_tensor(label_batch, device=self.device)
			emotion_tensor = float_tensor(emotion_batch, device=self.device)
			event_tensor = float_tensor(event_batch, device=self.device)

			ids_tensor_batch.append(cids_tensor)
			sgraph_tensor_batch.append(sgraph_tensor)
			clen_tensor_batch.append(clen_tensor)
			slen_tensor_batch.append(slen_tensor)
			distance_tensor_batch.append(distance_tensor)
			label_tensor_batch.append(label_tensor)
			emotion_tensor_batch.append(emotion_tensor)
			event_tensor_batch.append(event_tensor)
		return ids_tensor_batch, sgraph_tensor_batch, \
		clen_tensor_batch, slen_tensor_batch, distance_tensor_batch, label_tensor_batch, \
		emotion_tensor_batch, event_tensor_batch

	def _convert_sample2id(self, samples):
		ids, labels, clause_lens, sample_lens, emotions, events = [], [], [], [], [], []
		for sam in samples:
			sam_ids = []
			clen = []
			cemo = []
			for clause, emo in zip(sam[0], sam[2]):
				clause_ids = [self.word2id[word] for word in clause.split()][:self.max_sen_len]
				sam_ids.append(clause_ids)
				iclen = len(clause_ids)
				clen.append(iclen)
				cemo.append(emo)

			pairs = sam[1].split(', ')

			event = [0 for _ in range(len(sam[0]))]
			for eve in pairs:
				eve = eve.replace('(', '').replace(')', '').split(',')[1]
				event[int(eve)-1] = 1

			emotions.append(cemo)
			ids.append(sam_ids)	
			labels.append(pairs)
			clause_lens.append(clen)
			sample_lens.append(len(sam[0]))
			events.append(event)
		return ids, clause_lens, sample_lens, labels, emotions, events

	def load_pretrained_embedding(self, emb_dir, vocab_dir, word_list, dimension_size=200, tmp_dir='tmp'):
		tmp_emb_dir = os.path.join(tmp_dir, 'tmp_emb.npy')

		if not os.path.exists(tmp_emb_dir):
			pre_words = []
			with open(vocab_dir, 'r', encoding='utf-8') as fopen:
				for line in fopen:
					pre_words.append(line.strip())
			word2offset = {w: i for i, w in enumerate(pre_words)}
			word_vectors = []
			for i, word in enumerate(word_list):
				if word in word2offset:
					line = linecache.getline(emb_dir, word2offset[word]+2).strip()
					assert(word == line[:line.find(' ')])
					word_vectors.append(np.fromstring(line[line.find(' '):], sep=' ', dtype=np.float32))
				else:
					if i == 0:
						z = np.zeros(dimension_size, dtype=np.float32)
					else:
						z = np.random.rand(dimension_size) / 5. - 0.1
					word_vectors.append(z)
			word_vectors = np.stack(word_vectors)
			np.save(tmp_emb_dir, word_vectors)
		else:
			word_vectors = np.load(tmp_emb_dir)
		return word_vectors

	def load_vocab(self, data, tmp_dir='tmp'):
		vocab = set([])
		emotion = set([])
		max_clause_num = 0
		for index in data:
			text = data[index][0]
			for clause in text:
				vocab.update(clause.split(' '))
			if len(text) > max_clause_num:
				max_clause_num = len(text)
			cur_emotion = data[index][2]
			for emo in cur_emotion:
				emotion.update(emo.split('&'))
		vocab = sorted(list(vocab))
		vocab = ['<PAD>', '<UNK>'] + vocab
		emotion = sorted(list(emotion))
		emotion.remove('null')
		return vocab, emotion, max_clause_num

class CausePairRawData():
	def __init__(self, inp_files):
		self.data = self.load_data(inp_files['data_pair'])
		self.fold_index = self.load_fold_index(inp_files['fold_index'])

	def load_fold_index(self, inp_file):
		fold_index = defaultdict(list)
		with open(inp_file, 'r', encoding='utf-8') as f:
			for line in f.readlines():
				line = line.strip().split()
				head = int(line[0].split('_')[1])
				fold_index[head].append([int(z) for z in line[1:]])
		return fold_index

	def load_data(self, inp_file):
		data = {}
		with open(inp_file, 'r', encoding='utf-8') as f:
			lines = list(f.readlines())
			line_nums = len(lines)
			index = 0
			while index < line_nums:
				pair_index, pair_line_nums = [int(z) for z in lines[index].strip().split()]
				index += 1
				pair_label = lines[index].strip()
				text = []
				emotions = []
				for i in range(pair_line_nums):
					line = lines[index+i+1].strip().split(",")
					text.append(line[-1])
					emotions.append(line[1])
				assert (len(text) == pair_line_nums == len(emotions))
				index += pair_line_nums + 1
				data[pair_index] = (text, pair_label, emotions)
		return data

if __name__ == '__main__':
	inp_files = {
		'data_pair': '../data/splited_data/all_data_pair.txt', 
		'chains': '../data/splited_data/chains.txt', 
		'fold_index': '../data/splited_data/fold_index.txt',
		'tmp_dir': 'tmp',
		'emb_dir': '../data/w2v_200.txt',
		'vocab_dir': '../data/vocab.txt'
	}

	# test
	loader = CausePairLoader(inp_files, None)
	for i in range(1, 11):
		loader.generate_fold(i, 4, True, False)