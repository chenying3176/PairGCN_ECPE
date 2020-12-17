import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from .layers import DynamicAttLSTM, DynamicLSTM, GCN, GAT, weights_init_uniform, GCN

float_tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class STLSemGCNToyNet(nn.Module):
	def __init__(self, args, pretrained_emb):
		super().__init__()
		print('--------------------------')
		print("Model %s\t\t\t"%args.model)
		print("Input Dim %d\t\t"%args.emb_dim)
		print("Hidden Dim %d\t\t"%args.hidden_dim)
		print("RNN Num Layers %d\t"%args.num_layers)
		print('--------------------------')
		
		self.args = args
		self.text_emb = nn.Embedding.from_pretrained(
			float_tensor(pretrained_emb))
		self.distance_emb = nn.Embedding(4, 50)

		self.emotion_word_encoder = DynamicAttLSTM(args.emb_dim, 
			args.hidden_dim, args.lstm_dropout, True, args.num_layers, args.device)
		self.cause_word_encoder = DynamicAttLSTM(args.emb_dim, 
			args.hidden_dim, args.lstm_dropout, True, args.num_layers, args.device)

		self.emotion_clause_encoder = DynamicLSTM(args.hidden_dim*2, 
			args.hidden_dim, args.lstm_dropout, True, args.num_layers, args.device)
		self.event_clause_encoder = DynamicLSTM(args.hidden_dim*4, 
			args.hidden_dim, args.lstm_dropout, True, args.num_layers, args.device)

		self.gcn = GCN(args.hidden_dim*4, args.hidden_dim*2, args.dropout, args.num_gcn_layers)
		self.pair_decoder = nn.Linear(args.hidden_dim*2+50, 2)
		self.dropout = nn.Dropout(self.args.dropout)

	def extract_feature(self, x, text_rep, word_encoder, clause_encoder, emotion_inp=[None, None]):
		word_rep, alpha = word_encoder(text_rep, x['clen'])
		if emotion_inp[0] == None:
			inp = word_rep
		else:
			inp = torch.cat((word_rep, emotion_inp[1]), -1)
		seq = self.pad_clause_seq(inp, x['slen'])
		seq = self.dropout(seq)
		clause_rep = clause_encoder(seq, x['slen'])
		return clause_rep, alpha, word_rep

	def forward(self, x):
		text_rep = self.dropout(self.text_emb(x['text']))
		distance_rep = self.distance_emb(x['distance'])

		emotion_seq, _, emotion_inp = self.extract_feature(x, text_rep, self.emotion_word_encoder, self.emotion_clause_encoder)
		event_seq, _, _ = self.extract_feature(x, text_rep, self.cause_word_encoder, self.event_clause_encoder, [0, emotion_inp])

		clause_num = emotion_seq.size(1)
		pair_seq = torch.cat(
			(emotion_seq.unsqueeze(2).repeat(1, 1, clause_num, 1), 
			event_seq.unsqueeze(1).repeat(1, clause_num, 1, 1)), -1)
		# pair_seq = self.dropout(pair_seq)
		sem_pair_seq = self.gcn(x['sgraph'], pair_seq)
		# sem_pair_seq = self.gcn(pair_seq)
		sem_pair_seq = torch.cat((sem_pair_seq, distance_rep), -1)
		sem_pair_seq = self.dropout(sem_pair_seq)
		pair_logit = self.pair_decoder(sem_pair_seq)
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