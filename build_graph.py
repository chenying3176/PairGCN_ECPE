import json
def load_data(inp_file):
	with open(inp_file, 'r', encoding='utf-8') as f:
		data = json.load(f)
	return data

def build(docs):
	graphs = {}
	for doc_idx in docs:
		edges = set([])
		doc = docs[doc_idx]
		ssize = doc['ssize']
		clause_num = sum(ssize)
		clause_idx = list(range(clause_num))
		end_idx = [0]
		for s in ssize:
			end_idx.append(end_idx[-1]+s)
		end_idx = end_idx[1:]
		for i, end in enumerate(end_idx[:-1]):
			edges.add((end-1, end_idx[i+1]-1))
		end_idx = [0] + end_idx
		# for i, end in enumerate(end_idx[1:]):
		for i in range(1, len(end_idx)):
			start = end_idx[i-1]
			end = end_idx[i]
			nodes = list(range(start, end))
			for n in nodes[:-1]:
				edges.add((n, nodes[-1]))
		if len(edges) == 0:
			graphs[doc_idx] = 'null'
		else:
			graphs[doc_idx] = sorted(list(edges))
	return graphs

def main():
	docs = load_data('sent_dep/data/splited_data/doc_deps.json')
	graphs = build(docs)
	keys = sorted([int(z) for z in graphs.keys()])
	with open('data/splited_data/chains_sent.txt', 'w', encoding='utf-8') as f:
		for key in keys:
			f.write(str(key) + '\n')
			for edge in graphs[str(key)]:
				f.write(str(edge) + '\n')


if __name__ == '__main__':
	main()