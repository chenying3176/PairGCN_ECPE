from stanfordcorenlp import StanfordCoreNLP
import json
import os
from collections import Counter, defaultdict
import itertools

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def load_data(inp_file):
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
			for i in range(pair_line_nums):
				line = lines[index+i+1].strip().split(",")
				text.append(line[-1])
			assert (len(text) == pair_line_nums)
			index += pair_line_nums + 1
			data[pair_index] = (text, pair_label)
	return data

def load_pair(inp_file):
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
			for i in range(pair_line_nums):
				line = lines[index+i+1].strip()
				text.append(line)
			assert (len(text) == pair_line_nums)
			index += pair_line_nums + 1
			data[pair_index] = (text, pair_label)
	return data

def assign_token_clabel(tokens):
	labels = {}
	cur_label = 1
	for tok in tokens:
		if tok['word'] != ',':
			labels[tok['index']] = cur_label
		else:
			labels[tok['index']] = cur_label
			cur_label += 1
	return labels

def assign_coref_chain_clabel(coref, token_clabels):
	nodes = set([])
	for enti in coref[1]:
		entity_clabel_candidates = [token_clabels[tid] for tid in range(enti['startIndex'], enti['endIndex'])]
		entity_clabel = most_common(entity_clabel_candidates)
		nodes.add(entity_clabel)
	nodes = sorted(list(nodes))
	chains = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
	chains = set(chains)
	return chains

def assign_dep_clabel(dep, token_clabels):
	if dep['governorGloss'] == 'ROOT':
		return None
	governor = dep['governor']
	dependent = dep['dependent']
	return (token_clabels[governor], token_clabels[dependent])

def extract_coref(inp_folder):
	json_files = [f for f in os.listdir(inp_folder) if '.json' in f]
	samples = {}
	for i, jfile in enumerate(sorted(json_files)):
		sample_index = int(jfile.split('.')[0].split('_')[-1])
		inp_file = inp_folder + '/' + jfile
		with open(inp_file, 'r', encoding='utf-8') as f:
			json_data = json.load(f)
			sentence = json_data['sentences'][0]
			entity = sentence['entitymentions']
			tokens = sentence['tokens']
			corefs = json_data['corefs']
			chains = set([])
			token_clabels = assign_token_clabel(tokens)
			for coref in corefs:
				chain_clabels = assign_coref_chain_clabel([coref, corefs[coref]], token_clabels)
				chains.update(chain_clabels)
			chains = sorted(list(chains))
			samples[sample_index] = chains

	with open('data/splited_data/chains.txt', 'w', encoding='utf-8') as f:
		for sid in sorted(list(samples.keys())):
			f.write(str(sid) + '\n')
			if len(samples[sid]) == 0:
				f.write('null\n')
				continue
			for chain in samples[sid]:
				f.write(str(chain) + '\n')

def extract_dep(inp_folder):
	json_files = [f for f in os.listdir(inp_folder) if '.json' in f]
	samples = {}
	for i, jfile in enumerate(sorted(json_files)):
		sample_index = int(jfile.split('.')[0].split('_')[-1])
		inp_file = inp_folder + '/' + jfile
		with open(inp_file, 'r', encoding='utf-8') as f:
			json_data = json.load(f)
			sentence = json_data['sentences'][0]
			dependencies = sentence['enhancedPlusPlusDependencies']
			tokens = sentence['tokens']

			deps = set([])
			token_clabels = assign_token_clabel(tokens)
			for dep in dependencies:
				dep_clabels = assign_dep_clabel(dep, token_clabels)
				if dep_clabels == None:
					continue
				deps.add(dep_clabels)
			deps = sorted(list(deps))
			samples[sample_index] = deps

	with open('data/splited_data/deps.txt', 'w', encoding='utf-8') as f:
		for sid in sorted(list(samples.keys())):
			f.write(str(sid) + '\n')
			if len(samples[sid]) == 0:
				f.write('null\n')
				continue
			for chain in samples[sid]:
				f.write(str(chain) + '\n')

def fix_chains(inp_file='data/splited_data/chains.txt', out_file='data/splited_data/fixed_chains.txt'):
	def fix(chain):
		def unique(node, node_strs, idx):
			for i, n in enumerate(node_strs):
				if len(node) > len(n):
					continue
				if len(node.intersection(n)) == len(node) and idx != i:
					return i
			return idx

		def remove_dup(node_sets):
			unique_node_sets = set([])
			node_sets.sort(key=lambda x:len(x), reverse=True)
			for idx, node in enumerate(node_sets):
				new_idx = unique(node, node_sets, idx)
				if new_idx <= idx:
					unique_node_sets.add(new_idx)
			assert(len(unique_node_sets) > 0)
			return [node_sets[idx] for idx in unique_node_sets]

		def merge(node_sets):
			if len(node_sets) == 1:
				return node_sets
			while True:
				flag = True
				merged_node_sets = []
				for i in range(len(node_sets)):
					cur_node = node_sets[i]
					candi_nodes = [cur_node]
					for j in range(len(node_sets)):
						next_node = node_sets[j]
						inter = cur_node.intersection(next_node)
						if i == j or (len(inter) == len(cur_node) and len(next_node) == len(inter)):
							continue
						if len(inter) > 0:
							candi_nodes.append(next_node)
							flag = False
					new_node = set([])
					for node in candi_nodes:
						new_node = new_node.union(node)
					merged_node_sets.append(new_node)

				if flag:
					break
				else:
					node_sets = merged_node_sets
			return node_sets

		node_sets = []
		for c in chain:
			c = c.replace('(', '').replace(')', '').replace(' ', '').split(',')
			if len(node_sets) == 0:
				node_sets.append(set(c))

			pool = set(['_'.join(node) for node in node_sets])

			for z, node_set in enumerate(node_sets):
				c_set = set(c)
				if '_'.join(c_set) in pool:
					continue
				inter = c_set.intersection(node_set)
				if len(inter) > 0 and len(c_set) != len(node_set):
					node_sets[z] = node_set.union(c_set)
					pool.add('_'.join(node_sets[z]))
				elif len(inter) == 0:
					node_sets.append(c_set)
					pool.add('_'.join(c_set))

		if 'null' not in chain:
			node_sets = remove_dup(node_sets)
			node_sets = merge(node_sets)

		new_chains = []
		for node in node_sets:
			new_chain = ['('+', '.join(c)+')' for c in itertools.combinations(node, 2)]
			new_chains.extend(new_chain)
		return new_chains

	ori_chains = {}
	with open(inp_file, 'r', encoding='utf-8') as f:
		chain = []
		lines = f.readlines()
		idx = lines[0].strip()
		line_num = 1
		while line_num < len(lines):
			line = lines[line_num].strip()
			if line[0] != '(' and line != 'null':
				ori_chains[idx] = chain
				chain = []
				idx = line
			else:
				chain.append(line)
			line_num += 1
		if len(chain) > 0:
			ori_chains[idx] = chain

	fixed_chains = {}
	for idx in sorted(ori_chains):
		print('processing %s' % idx)
		if len(ori_chains[idx]) == 1 or ori_chains == 'null':
			z = ori_chains[idx]
		else:
			z = sorted(list(set(fix(ori_chains[idx]))))
		fixed_chains[idx] = z

	with open(out_file, 'w', encoding='utf-8') as f:
		for idx in sorted(map(int, fixed_chains.keys())):
			idx = str(idx)
			f.write(idx + '\n')
			for c in fixed_chains[idx]:
				f.write(c + '\n')


def extract_fold_index(inp_folder):
	index = []
	for i in range(1, 11):
		train_data = load_data(inp_folder + '/fold%d_train.txt'%i)
		test_data = load_data(inp_folder + '/fold%d_test.txt'%i)
		index.append(list(train_data.keys()))
		index.append(list(test_data.keys()))

	with open(inp_folder + '/fold_index.txt', 'w', encoding='utf-8') as f:
		for i, line in enumerate(index):
			head = 'train' if i % 2 == 0 else 'test'
			idx = i // 2 + 1
			head = head + '_' + str(idx)
			f.write(head + ' ' + ' '.join([str(z) for z in line]) + '\n')

def extract_vocab(inp_folder):

	vocab = []
	with open(inp_folder + '/w2v_200.txt', 'r', encoding='utf-8') as f:
		for i, line in enumerate(f.readlines()):
			if i == 0:
				continue
			line = line.strip().split(' ')
			if len(line) == 0:
				continue
			vocab.append(line[0])

	with open(inp_folder+'/vocab.txt', 'w', encoding='utf-8') as f:
		for word in vocab:
			f.write(word + '\n')

def extract_clause_json(inp_folder, all_clause, all_deps):
	def pop(tokens, deps):
		index = [token['index'] for token in tokens]
		words = [token['originalText'] for token in tokens]
		pos = [token['pos'] for token in tokens]
		ner = [token['ner'] for token in tokens]
		dep_ = [dep['dep'] for dep in deps]
		governor = [dep['governor'] for dep in deps]
		dependent = [dep['dependent'] for dep in deps]
		return index, words, pos, ner, dep_, governor, dependent

	inp_files = [inp_file for inp_file in os.listdir(inp_folder) if ('json' in inp_file and 'all_text' in inp_file)]
	for inp_file in inp_files:
		doc_idx, clause_idx = inp_file.split('_')[2:4]
		clause_idx = clause_idx.split('.')[0]
		doc_idx = int(doc_idx)
		clause_idx = int(clause_idx)
		with open(os.path.join(inp_folder, inp_file), 'r', encoding='utf-8') as f:
			clause_json = json.load(f)
			clause_tokens = clause_json['sentences'][0]['tokens']
			clause_tokens = ' '.join([token['originalText'] for token in clause_tokens])
			all_clause[doc_idx][clause_idx] = clause_tokens
			index, words, pos, ner, dep_, governor, dependent = pop(clause_json['sentences'][0]['tokens'], clause_json['sentences'][0]['enhancedPlusPlusDependencies'])
			clause_deps = {
				'pos': pos,
				'dep': dep_, 
				'governor': governor, 
				'dependent': dependent}
			all_deps[doc_idx][clause_idx] = clause_deps
	return all_clause, all_deps

def convert_pair(pairs, all_clause, chain_map, output_file):
	new_seg = defaultdict(list)
	label = {}
	for pair_id in pairs:
		clause = all_clause[chain_map[pair_id]]
		pair = pairs[pair_id]
		clause_size = len(pair[0])
		assert(len(clause) == len(pair[0]))
		for i in range(clause_size):
			cur_clause = pair[0][i].split(',')[:-1] + [clause[i]]
			new_seg[pair_id].append(','.join(cur_clause))
		label[pair_id] = pair[1]
		assert(len(clause) == len(new_seg[pair_id]))

	with open(output_file, 'w', encoding='utf-8') as f:
		for pair_id in sorted(list(new_seg)):
			p = label[pair_id]
			clauses = new_seg[pair_id]
			f.write('%d %d'%(pair_id, len(clauses))+'\n')
			f.write(' '+p + '\n')
			for c in clauses:
				f.write(c+'\n')

def main():

	inp_file = 'data/splited_data/all_data_pair.txt'
	pairs = load_pair(inp_file)
	chain_map = {int(index):int(i) for i, index in enumerate(list(pairs.keys()))}
	all_clause = defaultdict(dict)
	all_deps = defaultdict(dict)
	for i in range(1, 13):
		inp_folder = 'data/clause_data_%d'%i
		all_clause, all_deps = extract_clause_json(inp_folder, all_clause, all_deps)

	with open('data/splited_data/clause_deps.json', 'w', encoding='utf-8') as f:
		json.dump(all_deps, f, indent=4, ensure_ascii=False, sort_keys=True)
	convert_pair(pairs, all_clause, chain_map, 'data/splited_data/all_data_pair_new.txt')

if __name__ == '__main__':
	main()