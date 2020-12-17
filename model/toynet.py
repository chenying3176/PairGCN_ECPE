import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from .layers import DynamicAttLSTM, DynamicLSTM, weights_init_uniform

float_tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class ToyNet(nn.Module):
	def __init__(self, args, pretrained_emb):
		super().__init__()
		print('--------------------------')
		print(args.model)
		print('--------------------------')
		self.args = args
		self.text_emb = nn.Embedding.from_pretrained(
			float_tensor(pretrained_emb))
		self.distance_emb = nn.Embedding(args.max_doc_len+1, args.hidden_dim)

		self.emotion_word_encoder = DynamicAttLSTM(args.emb_dim, 
			args.hidden_dim, args.lstm_dropout, True, args.num_layers, args.device)
		self.cause_word_encoder = DynamicAttLSTM(args.emb_dim, 
			args.hidden_dim, args.lstm_dropout, True, args.num_layers, args.device)

		self.emotion_clause_encoder = DynamicLSTM(args.hidden_dim*2, 
			args.hidden_dim, args.lstm_dropout, True, args.num_layers, args.device)
		self.event_clause_encoder = DynamicLSTM(args.hidden_dim*2, 
			args.hidden_dim, args.lstm_dropout, True, args.num_layers, args.device)

		self.emotion_gcn = GCN(args.hidden_dim*2, args.hidden_dim*2, 0.5, 2)
		self.event_gcn = GCN(args.hidden_dim*2, args.hidden_dim*2, 0.5, 2)

		self.pair_mlp = nn.Linear(args.hidden_dim*8, args.hidden_dim*4)
		self.pair_decoder = nn.Linear(args.hidden_dim*5, 2)
		self.emotion_decoder = nn.Linear(args.hidden_dim*4, 2)
		self.event_decoder = nn.Linear(args.hidden_dim*4, 2)

		self.dropout = nn.Dropout(0.5)
		self.apply(weights_init_uniform)

	def forward(self, x):
		text_rep = self.text_emb(x['text'])
		text_rep = self.dropout(text_rep)
		distance_rep = self.distance_emb(x['distance'])

		emotion_rep, emotion_alpha = self.emotion_word_encoder(text_rep, x['clen'])
		event_rep, event_alpha = self.cause_word_encoder(text_rep, x['clen'])

		emotion_seq = self.pad_clause_seq(emotion_rep, x['slen'])
		emotion_seq = self.dropout(emotion_seq)
		emotion_seq = self.emotion_clause_encoder(emotion_seq, x['slen'])
		emotion_con = self.emotion_gcn(x['chain'], self.dropout(emotion_seq))
		emotion_seq = torch.cat((emotion_con, emotion_seq), -1)
		emotion_seq = self.dropout(emotion_seq)

		event_seq = self.pad_clause_seq(event_rep, x['slen'])
		event_seq = self.dropout(event_seq)
		event_seq = self.event_clause_encoder(event_seq, x['slen'])
		event_con = self.event_gcn(x['chain'], self.dropout(event_seq))
		event_seq = torch.cat((event_con, event_seq), -1)
		event_seq = self.dropout(event_seq)

		pair_seq = torch.cat(
			(emotion_seq.unsqueeze(2).repeat(1, 1, emotion_seq.size(1), 1), 
			event_seq.unsqueeze(1).repeat(1, event_seq.size(1), 1, 1)), -1)

		pair_seq = torch.cat((self.pair_mlp(pair_seq), distance_rep), -1)
		pair_seq = self.dropout(pair_seq)

		pair_logit = self.pair_decoder(pair_seq)
		# for emotion detection
		emotion_dete_inp = self.recover_clause(emotion_seq, x['slen'])
		emotion_logit = self.emotion_decoder(emotion_dete_inp).squeeze()

		event_dete_inp = self.recover_clause(event_seq, x['slen'])
		event_logit = self.event_decoder(event_dete_inp).squeeze()		

		return pair_logit, emotion_logit, event_logit, (emotion_alpha, event_alpha)

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
