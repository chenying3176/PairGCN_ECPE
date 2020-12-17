import sys 
sys.path.append("..")
from utils.data_loader import CausePairLoader
from model.toynet_semgcn_stl import STLSemGCNToyNet
import argparse
import torch
from collections import defaultdict
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_pair', type=str, default='data/splited_data/all_data_pair.txt')
parser.add_argument('--chains', type=str, default='data/splited_data/chains.txt')
parser.add_argument('--chain_map', type=str, default='data/splited_data/chain_map.txt')
parser.add_argument('--clause_dep', type=str, default='data/splited_data/doc_deps.json')
parser.add_argument('--fold_index', type=str, default='data/splited_data/fold_index.txt')
parser.add_argument('--tmp_dir', type=str, default='utils/tmp')
parser.add_argument('--emb_dir', type=str, default='data/w2v_200.txt')
parser.add_argument('--vocab_dir', type=str, default='data/vocab.txt')

parser.add_argument('--model', type=str, default='no_context_toynet_v1')

parser.add_argument('--batch_size', type=int, default=2, help='Training batch size.')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate.')
parser.add_argument('--l2', type=float, default=1e-8)
parser.add_argument('--num_epoch', type=int, default=30, help='Training batch size.')

# LSTM params
parser.add_argument('--emb_dim', type=int, default=200, help='Word embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=100, help='LSTM hidden dim.')
parser.add_argument('--lstm_dropout', type=float, default=0.5, help='LSTM dropout rate.')
parser.add_argument('--num_layers', type=int, default=1, help='Num of GCN layers.')
parser.add_argument('--num_gcn_layers', type=int, default=2, help='Num of GCN layers.')
parser.add_argument('--dropout', type=float, default=0.5, help='Num of GCN layers.')

parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
parser.add_argument('--lower', default=True, help='Lowercase all words.')
parser.add_argument('--direct', default=False)
parser.add_argument('--loop', default=True)

parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')

parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adamax', help='Optimizer: sgd, adagrad, adam or adamax.')

parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--seed', type=int, default=2020)


def inference(model, test_loader, loader):
	labels = []
	preds = []
	probas = []
	indexs = []
	types = []
	emotion_alphas = []
	event_alphas = []
	with torch.no_grad():
		for batch_id, batch in enumerate(zip(*test_loader)):
			model.eval()
			x = {
					'text': batch[0], 'sgraph': batch[1],
					'clen': batch[2], 'slen': batch[3], 
					'distance': batch[4] 
			}
			if batch_id == 28:
				print([loader.id2word[z.item()] for z in batch[0][9]])
				pred = model(x)
				pred = torch.softmax(pred, -1)
				print(pred.topk(1)[1][1, 2, 1])
				print(pred[1, 2, 1, 1])
				print(pred.size())
				exit(0)
			# if batch_id > 2:
			# 	exit(0)


def main():
	args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	loader = CausePairLoader(inp_files, args.device)
	labels, preds, probas, indexs, types, emotion_alphas, event_alphas = [], [], [], [], [], [], []
	for fold in range(1, 11):
		print(fold)
		_, test_loader = loader.generate_fold_data(fold, args.batch_size, False, False)
		model = STLSemGCNToyNet(args, loader.emb)
		checkpoint = torch.load('saved_models/stl_semgcn_toynet_fold1', map_location=args.device)
		model.load_state_dict(checkpoint['model'])
		inference(model, test_loader, loader)

if __name__ == '__main__':
	args = parser.parse_args()
	args.max_doc_len = 75

	inp_files = {
		'max_doc_len': 75, 
		'max_sen_len': 100,
		'data_pair': args.data_pair, 
		'chains': args.chains,
		'chain_map': args.chain_map, 
		'fold_index': args.fold_index,
		'tmp_dir': args.tmp_dir,
		'emb_dir': args.emb_dir,
		'vocab_dir': args.vocab_dir,
		'clause_dep': args.clause_dep
	}
	main()