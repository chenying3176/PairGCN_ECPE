import sys
sys.path.append('../')
import numpy as np
import torch
from model.dsgcn_pipeline import Pipeline
from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertModel,
    BertTokenizer,
)

def get_model(device):
	config_class, model_class, tokenizer_class = BertConfig, BertModel, BertTokenizer
	config = config_class.from_pretrained(
		'bert-base-chinese',
		num_labels=2,
		cache_dir=None,
		output_hidden_states=True,
		)
	tokenizer = tokenizer_class.from_pretrained(
		'bert-base-chinese',
		cache_dir=None,
		)
	model = model_class.from_pretrained(
		'bert-base-chinese',
		from_tf=False,
		config=config,
		cache_dir=None,
	)

	model.to(device)
	return model, tokenizer, config

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

def extract_features():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model, tokenizer, config = get_model(device)
	nlp = Pipeline(
		model=model, 
		tokenizer=tokenizer, 
		device=device)
	data = load_data('data/splited_data/all_data_pair.txt')
	bert_feat = {}
	part = 0
	for key in data:
		print(key)
		doc, label = data[key]
		for i in range(len(doc)):
			clause = doc[i].replace(' ', '')
			feature = nlp(clause)
			bert_feat[clause] = feature
			if len(bert_feat) >= 300:
				torch.save(bert_feat, 'data/all_bert_feat%d.npy'%part)
				part += 1
				bert_feat = {}
	if len(bert_feat) > 0:
		torch.save(bert_feat, 'data/all_bert_feat%d.npy'%part)


def main():
	extract_features()

if __name__ == '__main__':
	main()
