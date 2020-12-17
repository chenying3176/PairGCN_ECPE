import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from .layers import DynamicAttLSTM, DynamicLSTM, DSGCN, weights_init_uniform

float_tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
long_tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

class STLSemGCNToyNet(nn.Module):
	def __init__(self, args):
		super().__init__()
		print('--------------------------')
		print("Model %s\t\t\t"%args.model)
		print("Input Dim %d\t\t"%args.emb_dim)
		print("Hidden Dim %d\t\t"%args.hidden_dim)
		print("RNN Num Layers %d\t"%args.num_layers)
		print('--------------------------')
		
		self.args = args
		self.text_emb = nn.Linear(768, args.emb_dim)

		self.distance_emb = nn.Embedding(args.max_doc_len+1, args.hidden_dim)

		self.emotion_word_encoder = DynamicAttLSTM(args.emb_dim, 
			args.hidden_dim, args.lstm_dropout, True, args.num_layers, args.device)
		self.cause_word_encoder = DynamicAttLSTM(args.emb_dim, 
			args.hidden_dim, args.lstm_dropout, True, args.num_layers, args.device)

		self.emotion_clause_encoder = DynamicLSTM(args.hidden_dim*2, 
			args.hidden_dim, args.lstm_dropout, True, args.num_layers, args.device)
		self.event_clause_encoder = DynamicLSTM(args.hidden_dim*2, 
			args.hidden_dim, args.lstm_dropout, True, args.num_layers, args.device)

		self.sem_gcn = DSGCN(args.hidden_dim*2, args.hidden_dim*2, args.dropout, args.num_gcn_layers)
		self.pair_decoder = nn.Linear(args.hidden_dim*5, 2)

		self.dropout = nn.Dropout(args.dropout)
		self.apply(weights_init_uniform)

	def extract_feature(self, x, text_rep, word_encoder, clause_encoder, emotion_inp=[None, None]):
		word_rep, alpha = word_encoder(text_rep, x['clen'])
		if emotion_inp[0] == None:
			inp = word_rep
		else:
			inp = torch.cat((word_rep, emotion_inp[1]), -1)
		seq = self.pad_clause_seq(inp, x['slen'])
		seq = self.dropout(seq)
		clause_rep = clause_encoder(seq, x['slen'])
		clause_rep = self.dropout(clause_rep)
		return clause_rep, alpha, word_rep

	def forward(self, x):
		text_rep = self.dropout(self.text_emb(x['text']))
		distance_rep = self.distance_emb(x['distance'])
		emotion_rep, _ = self.emotion_word_encoder(text_rep, x['clen'])
		emotion_rep = self.pad_clause_seq(emotion_rep, x['slen'])
		emotion_rep = self.dropout(emotion_rep)

		cause_rep, _ = self.cause_word_encoder(text_rep, x['clen'])
		cause_rep = self.pad_clause_seq(cause_rep, x['slen'])
		cause_rep = self.dropout(cause_rep)

		adj = x['sgraph']
		split_point = adj.size(1) // 2
		sem_rep = self.sem_gcn(adj, torch.cat((emotion_rep, cause_rep), 1), x['slen'])
		sem_rep = self.dropout(sem_rep)
		sem_emotion_seq, sem_event_seq = sem_rep[:, :split_point], sem_rep[:, split_point:]

		sem_emotion_seq = self.emotion_clause_encoder(sem_emotion_seq, x['slen'])
		sem_emotion_seq = self.dropout(sem_emotion_seq)
		sem_event_seq = self.event_clause_encoder(sem_event_seq, x['slen'])
		sem_event_seq = self.dropout(sem_event_seq)

		pair_seq = torch.cat(
			(sem_emotion_seq.unsqueeze(2).repeat(1, 1, sem_emotion_seq.size(1), 1), 
			sem_event_seq.unsqueeze(1).repeat(1, sem_event_seq.size(1), 1, 1), 
			distance_rep), -1)

		pair_seq = self.dropout(pair_seq)
		pair_logit = self.pair_decoder(pair_seq)
		return pair_logit

	def pad_clause_seq(self, clause_seq, seq_len):
		clauses = []
		acc_len = 0		
		for sl in seq_len:
			clauses.append(clause_seq[acc_len:acc_len+sl])
			acc_len += sl
		pad_clauses = pad_sequence(clauses, batch_first=True)
		return pad_clauses

	def recover_clause(self, pad_clauses, seq_len):
		clauses = []
		for i, sl in enumerate(seq_len):
			clauses.append(pad_clauses[i, :sl])
		clauses = torch.cat(clauses, 0)
		return clauses