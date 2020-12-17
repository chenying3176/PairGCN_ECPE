import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prf_fn
from model.toynet_semgcn_stl_bert import STLSemGCNToyNet


class Wrapper():
	def __init__(self, args, loader):
		self.args = args
		self.loader, self.fold = loader
		self.train_loader, self.dev_loader, self.test_loader = self.loader.generate_fold_data(self.fold, args.batch_size, False, False)
		
		self.args.id2word = self.loader.id2word
		if args.model == 'stl_semgcn_toynet':
			self.model = STLSemGCNToyNet(args)

		self.model.to(args.device)
		self.parameters = [p for p in self.model.parameters() if p.requires_grad]
		self.optimizer = optim.Adam(self.parameters, lr=args.lr, weight_decay=args.l2)
		self.loss_fn = nn.CrossEntropyLoss(reduction='none')

	def run(self):
		dev_best_pair_prf = None
		best_pair_prf, best_emo_prf, best_cause_prf = None, [0, 0, 0], [0, 0, 0]
		for epoch in range(self.args.num_epoch):
			epoch_loss = self.train(epoch)
			dev_pair_prf, dev_emotion_prf, dev_cause_prf = self.evaluate(self.dev_loader)
			test_pair_prf, test_emotion_prf, test_cause_prf = self.evaluate(self.test_loader)

			print('Dev Epoch Pair\t%d R %0.4f P %0.4f F %0.4f' % (epoch, *dev_pair_prf))
			print('Test Epoch Pair\t%d R %0.4f P %0.4f F %0.4f' % (epoch, *test_pair_prf))
			print('Test Epoch Emotion\t%d R %0.4f P %0.4f F %0.4f' % (epoch, *test_emotion_prf))
			print('Test Epoch Cause\t%d R %0.4f P %0.4f F %0.4f' % (epoch, *test_cause_prf))
			print('---------Epoch End----------')
			if dev_best_pair_prf == None or dev_pair_prf[-1] > dev_best_pair_prf[-1]:
				dev_best_pair_prf = dev_pair_prf
				best_pair_prf = test_pair_prf
				best_emo_prf = test_emotion_prf
				best_cause_prf = test_cause_prf

		print('Dev Pair\t%d R %0.4f P %0.4f F %0.4f' % (self.fold, *dev_best_pair_prf))
		print('Fold Pair\t%d R %0.4f P %0.4f F %0.4f' % (self.fold, *best_pair_prf))
		print('Fold Emotion\t%d R %0.4f P %0.4f F %0.4f' % (self.fold, *best_emo_prf))
		print('Fold Cause\t%d R %0.4f P %0.4f F %0.4f' % (self.fold, *best_cause_prf))
		print('-----------Fold End-----------')
		print()
		return best_pair_prf, best_emo_prf, best_cause_prf, dev_best_pair_prf

	def train(self, epoch):
		epoch_loss = 0.0
		with tqdm(total=len(self.train_loader[0])) as pbar:
			for batch_id, batch in enumerate(zip(*self.train_loader)):
				x = {
					'text': batch[0], 'sgraph': batch[1],
					'clen': batch[2], 'slen': batch[3], 
					'distance': batch[4] 
				}
				y_true = batch[-3:]
				batch_loss = self.update(x, y_true, batch[3], batch[4], epoch)
				epoch_loss += batch_loss
				pbar.set_description("Epoch %d\tLoss %0.4f" % (epoch, batch_loss))
				pbar.update()
		return epoch_loss

	def update(self, x, y_true, clen, slen, epoch):
		self.model.train()
		y_pair_true, y_emotion_true, y_cause_true = y_true
		y_pair_pred = self.model(x)
		y_pair_mask = (y_pair_true != -1).long()
		y_pair_true = y_pair_mask * y_pair_true.long()
		y_pair_mask = y_pair_mask.reshape(-1)
		y_pair_true = y_pair_true.reshape(-1)
		y_pair_pred = y_pair_pred.reshape(-1, 2)
		pair_loss = self.loss_fn(y_pair_pred, y_pair_mask * y_pair_true.long()) * y_pair_mask
		loss = pair_loss.sum() / y_pair_mask.sum()

		self.model.zero_grad()
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item()

	def evaluate(self, data_loader):
		y_pair = [[], []]
		y_emotion = [[], []]
		y_cause = [[], []]

		true_pairs = []
		pred_pairs = []
		with torch.no_grad():
			for batch_id, batch in enumerate(zip(*data_loader)):
				self.model.eval()
				x = {
					'text': batch[0], 'sgraph': batch[1],
					'clen': batch[2], 'slen': batch[3], 
					'distance': batch[4] 
				}
				y_pair_true = batch[-3]
				y_pair_pred = self.model(x)

				true_pairs.append(y_pair_true.cpu().numpy())
				pred_pairs.append(y_pair_pred.topk(1)[1].squeeze().cpu().numpy())

				y_pair_pred = y_pair_pred.reshape(-1, 2).topk(1)[1].squeeze().cpu().numpy()
				y_pair_true = y_pair_true.reshape(-1).cpu().numpy()
				for y_pred, y_true in zip(y_pair_pred, y_pair_true):
					if y_true != -1:
						y_pair[0].append(y_true)
						y_pair[1].append(y_pred)
		pair_prf = prf_fn(y_pair[0], y_pair[1], average='binary')[:-1]
		emotion_prf, cause_prf = self.evaluate_subtask(true_pairs, pred_pairs)
		return pair_prf, emotion_prf, cause_prf

	def evaluate_subtask(self, true_pairs, pred_pairs):
		emotion_pred, emotion_true = [], []
		cause_pred, cause_true = [], []
		for tpair, ppair in zip(true_pairs, pred_pairs):
			mask = tpair != -1
			size = (tpair[:, 0] != -1).sum(-1)
			tpair = mask * tpair
			ppair = mask * ppair
			temotion = [a[:s].astype(int) for s, a in zip(size, (tpair.sum(2) > 0))]
			tcause = [a[:s].astype(int) for s, a in zip(size, (tpair.sum(1) > 0))]
			pemotion = [a[:s].astype(int) for s, a in zip(size, (ppair.sum(2) > 0))]
			pcause = [a[:s].astype(int) for s, a in zip(size, (ppair.sum(1) > 0))]
			for a, b, c, d in zip(temotion, pemotion, tcause, pcause):
				emotion_true.extend(a)
				emotion_pred.extend(b)
				cause_true.extend(c)
				cause_pred.extend(d)
		emotion_prf = prf_fn(emotion_true, emotion_pred, average='binary')[:-1]
		cause_prf = prf_fn(cause_true, cause_pred, average='binary')[:-1]
		return emotion_prf, cause_prf

	def save(self, filename):
		params = {
			'model': self.model.state_dict(),
			'config': self.args,
			}
		try:
			output_path = os.path.join(self.args.save_dir, filename + '_fold%d'%self.fold)
			torch.save(params, output_path)
			print("model saved to {}".format(filename))
		except BaseException:
			print("[Warning: Saving failed... continuing anyway.]")

	def load(self, filename):
		try:
			checkpoint = torch.load(filename)
		except BaseException:
			print("Cannot load model from {}".format(filename))
			exit()
		self.model.load_state_dict(checkpoint['model'])
		self.args = checkpoint['config']