import torch
import argparse
import numpy as np
import random
from utils.data_loader import CausePairLoader
from wrapper import Wrapper

parser = argparse.ArgumentParser()
# data source
parser.add_argument('--data_pair', type=str, default='data/splited_data/all_data_pair.txt')
parser.add_argument('--fold_index', type=str, default='data/splited_data/new_fold_index.txt')
parser.add_argument('--tmp_dir', type=str, default='utils/tmp')
parser.add_argument('--emb_dir', type=str, default='data/w2v_200.txt')
parser.add_argument('--vocab_dir', type=str, default='data/vocab.txt')

parser.add_argument('--model', type=str, default='stl_semgcn_toynet')
# optimizer params
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate.')
parser.add_argument('--l2', type=float, default=0.0)
parser.add_argument('--num_epoch', type=int, default=30, help='Training batch size.')

# model params
parser.add_argument('--emb_dim', type=int, default=200, help='Word embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=100, help='LSTM hidden dim.')
parser.add_argument('--num_layers', type=int, default=1, help='Num of GCN layers.')
parser.add_argument('--num_gcn_layers', type=int, default=2, help='Num of GCN layers.')
parser.add_argument('--dropout', type=float, default=0.5, help='Num of GCN layers.')
parser.add_argument('--lstm_dropout', type=float, default=0.5, help='Num of GCN layers.')

parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--repeat', type=int, default=2)

def seeds(seed):
	# to ensure reproducibility
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def main():
	avg_pair_results = []
	avg_emotion_results = []
	avg_event_results = []

	dev_avg_result = []
	total_fold = 10

	for fold in range(1, total_fold+1):
		model_wrapper = Wrapper(args, (loader, fold))
		fold_result = model_wrapper.run()
		avg_pair_results.append(fold_result[0])
		avg_emotion_results.append(fold_result[1])
		avg_event_results.append(fold_result[2])
			
		dev_avg_result.append(fold_result[3])

		acc_pair_prf = avg_fn(avg_pair_results, 0), avg_fn(avg_pair_results, 1), avg_fn(avg_pair_results, 2)
		acc_emotion_prf = avg_fn(avg_emotion_results, 0), avg_fn(avg_emotion_results, 1), avg_fn(avg_emotion_results, 2)
		acc_event_prf = avg_fn(avg_event_results, 0), avg_fn(avg_event_results, 1), avg_fn(avg_event_results, 2)

		print('Acc Fold Pair\t%d R %0.4f P %0.4f F %0.4f' % (fold, *acc_pair_prf))
		print('Acc Fold Emoti\t%d R %0.4f P %0.4f F %0.4f' % (fold, *acc_emotion_prf))
		print('Acc Fold Event\t%d R %0.4f P %0.4f F %0.4f' % (fold, *acc_event_prf))
		print('-----------------------------')
		print()
	print('------------Final Result-------------')
	dev_pair_prf = avg_fn(dev_avg_result, 0), avg_fn(dev_avg_result, 1), avg_fn(dev_avg_result, 2)
	
	pair_prf = avg_fn(avg_pair_results, 0), avg_fn(avg_pair_results, 1), avg_fn(avg_pair_results, 2)
	emotion_prf = avg_fn(avg_emotion_results, 0), avg_fn(avg_emotion_results, 1), avg_fn(avg_emotion_results, 2)
	event_prf = avg_fn(avg_event_results, 0), avg_fn(avg_event_results, 1), avg_fn(avg_event_results, 2)
	print('Dev Pair\t R %0.4f P %0.4f F %0.4f' % dev_pair_prf)
	print('Final Pair\t R %0.4f P %0.4f F %0.4f' % pair_prf)
	print('Final Emotion\t R %0.4f P %0.4f F %0.4f' % emotion_prf)
	print('Final Event\t R %0.4f P %0.4f F %0.4f' % event_prf)
	print('---------------Finish----------------')
	return pair_prf, emotion_prf, event_prf, dev_pair_prf

if __name__ == '__main__':
	avg_fn = lambda x, index: sum([z[index] for z in x]) / len(x)

	pair_results = []
	emotion_results = []
	event_results = []
	dev_pair_results = []

	args = parser.parse_args()

	args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	args.max_doc_len = 75
	args.max_sen_len = 100
	inp_files = {
		'max_doc_len': args.max_doc_len, 
		'max_sen_len': args.max_sen_len,
		'data_pair': args.data_pair, 
		'fold_index': args.fold_index,
		'tmp_dir': args.tmp_dir,
		'emb_dir': args.emb_dir,
		'vocab_dir': args.vocab_dir,
	}
	loader = CausePairLoader(inp_files, args.device)
	for i in range(args.repeat):
		print('Repeat %d, Seed %d' % (i, args.seed+i))
		seeds(args.seed+i)
		args.repeat = i
		pair_prf, emotion_prf, event_prf, dev_pair_prf = main()
		pair_results.append(pair_prf)
		emotion_results.append(emotion_prf)
		event_results.append(event_prf)
		dev_pair_results.append(dev_pair_prf)
	repeat_pair_prf = avg_fn(pair_results, 0), avg_fn(pair_results, 1), avg_fn(pair_results, 2)
	repeat_emotion_prf = avg_fn(emotion_results, 0), avg_fn(emotion_results, 1), avg_fn(emotion_results, 2)
	repeat_event_prf = avg_fn(event_results, 0), avg_fn(event_results, 1), avg_fn(event_results, 2)
	repeat_dev_pair_prf = avg_fn(dev_pair_results, 0), avg_fn(dev_pair_results, 1), avg_fn(dev_pair_results, 2)
	print('Dev Pair\t R %0.4f P %0.4f F %0.4f' % repeat_dev_pair_prf)
	print('Final Pair\t R %0.4f P %0.4f F %0.4f' % repeat_pair_prf)
	print('Final Emotion\t R %0.4f P %0.4f F %0.4f' % repeat_emotion_prf)
	print('Final Event\t R %0.4f P %0.4f F %0.4f' % repeat_event_prf)
	print('----------Repeat Finish----------------')
