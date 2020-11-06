import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

from time import time

MAX_VOCAB = 10000
batch_size = 20

class LSTMCell(nn.Module):

	def __init__(self, input_size, hidden_size):
		super(LSTMCell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=True)
		self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
		self.reset_parameters()

	def reset_parameters(self):
		std = 1.0 / math.sqrt(self.hidden_size)
		for w in self.parameters():
			w.data.uniform_(-std, std)

	def forward(self, x, hidden):
		#import pdb; pdb.set_trace()
		hx, cx = hidden
		x = x.view(-1, x.size(1))
		gates = self.x2h(x) + self.h2h(hx)
		gates = gates.squeeze()
		
		ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

		ingate = torch.sigmoid(ingate)
		forgetgate = torch.sigmoid(forgetgate)
		cellgate = torch.tanh(cellgate)
		outgate = torch.sigmoid(outgate)
		
		cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
		hy = torch.mul(outgate, torch.tanh(cy))

		return (hy, cy)

vocab = {}
def load_data(file):
	vocab_idx = 0
	with open(file) as f:
		text = f.read().replace("\n","<eos>")
	arr = text.split()
	data = np.zeros(len(arr), dtype='int32')
	for i, word in enumerate(arr):
		if word not in vocab:
			vocab[word] = vocab_idx
			vocab_idx = vocab_idx + 1
		data[i] = vocab[word]
	return batcherize(np.array(data), batch_size)

def batcherize(corpus, batch_size):
	s = len(corpus)
	x = np.zeros((batch_size, s // batch_size), dtype='int32')
	start = 0
	for i in range(batch_size):
		finish = start + x.shape[1]
		x[i,:] = corpus[start:finish]
		start = finish
	return x

class RNN(nn.Module):
	def __init__(self, vocab_size, input_dim, hidden_dim, n_layers, dropouts, init_scale):
		super(RNN, self).__init__()
		self.vocab_size = vocab_size
		self.input_dim = hidden_dim
		self.hidden_dim = hidden_dim
		self.n_layers = n_layers
		self.dropouts = dropouts
		self.initrange = init_scale
		
		self.encoder = nn.Embedding(vocab_size, input_dim)
		self.drop = nn.Dropout(dropouts)
		self.rnn1 = LSTMCell(input_dim, hidden_dim)
		self.rnn2 = LSTMCell(input_dim, hidden_dim)
		self.decoder = nn.Linear(hidden_dim, vocab_size)

	def forward(self, input, hidden):
		#import pdb;pdb.set_trace()
		emb = self.drop(self.encoder(input))
		out1 = torch.zeros_like(emb)
		hx, cx = hidden[0]
		for step in range(emb.shape[1]):
			hx, cx = self.rnn1(emb[:,step], (hx, cx))
			out1[:, step] = hx
		out1 = self.drop(out1)
		out2 = torch.zeros_like(emb)
		hx2, cx2 = hidden[1]
		for step in range(emb.shape[1]):
			hx2, cx2 = self.rnn2(out1[:,step], (hx2, cx2))
			out2[:, step] = hx2
		out2 = self.drop(out2)
		output = self.decoder(out2)
		return output, ((hx, cx), (hx2, cx2))

	def init_weights(self):
		initrange = self.initrange
		self.encoder.weight.data.uniform_(-initrange,initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange,initrange)

	def init_hidden(self, batch_size):
		return (((Variable(torch.zeros(batch_size, self.hidden_dim)),) * 2),) * self.n_layers

def create_model(train_params):
	print(train_params)
	return RNN(MAX_VOCAB, train_params['hidden_size'], train_params['hidden_size'], 2, 
		train_params['dropout'], train_params['init_scale'])

def repackage_hidden(h):
	if isinstance(h, torch.Tensor):
		return h.detach()
	else:
		return tuple(repackage_hidden(v) for v in h)

def train_model(model, corpus, criterion, train_params, valid, test):
	win_size = int(train_params['win_size'])
	epochs = int(train_params['epochs'])
	epoch_size = train.shape[1] // win_size
	print("epoch_size", epoch_size)
	lr_decayed = train_params['learning_rate']
	for cnt in range(epochs):
		if train_params['epoch_decay_start'] <= cnt:
			lr_decayed /= (train_params['decay'] * 1.0)
		model.train()
		optimizer = torch.optim.SGD(model.parameters(), lr=lr_decayed)
		partial_loss = 0.0
		hidden = model.init_hidden(batch_size)
		for count_v, offset in enumerate(range(0, corpus.shape[1] - 1, win_size)):
			seq_len = int(min(win_size, corpus.shape[1] - offset - 1))
			
			X = torch.LongTensor(corpus[:,offset     : offset + seq_len    ])
			Y = torch.LongTensor(corpus[:,offset + 1 : offset + seq_len + 1])
			
			hidden = repackage_hidden(hidden)
			#import pdb; pdb.set_trace()
			# forward pass
			output, hidden = model(X, hidden)
			loss = criterion(output.view(-1, MAX_VOCAB), Y.reshape(-1))
			
			# partial loss
			partial_loss += loss.item()
			
			# zero loss
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), train_params['clip_norm'])
			optimizer.step()
			
			if count_v % (epoch_size // 10) == 50:
				loss = partial_loss / ((epoch_size // 10) * seq_len * batch_size * 1.0)
				print("Iteration {:3.2f}, Offset {:5d}, loss={:6.5f}, perplexity={:6.5f}, lr={:3.4f}".format(cnt + count_v / epoch_size, offset, loss, math.exp(loss), lr_decayed))
				partial_loss = 0.0
		loss = test_model(model, train, criterion, default_params)
		loss = test_model(model, valid, criterion, default_params)
		loss = test_model(model, test, criterion, default_params)

def test_model(model, corpus, criterion, params):
	model.eval()
	win_size = int(param['win_size'])
	avg = 0.0
	sum_v = 0.0
	count_v = 0.0
	offset = 0
	hidden = model.init_hidden(batch_size)
	losses = 0.0
	
	for count_v, offset in enumerate(range(0, corpus.shape[1] - 1, win_size)):
		seq_len = min(win_size, corpus.shape[1] - offset - 1)
		X = torch.LongTensor(corpus[:,offset     : offset + seq_len    ])
		Y = torch.LongTensor(corpus[:,offset + 1 : offset + seq_len + 1])
		hidden = repackage_hidden(hidden)
		output, hidden = model(X, hidden)
		loss = criterion(output.view(-1, MAX_VOCAB), Y.reshape(-1))
		avg += loss.item()
		sum_v += np.prod(Y.size())
	avg = avg / float(sum_v)
	print("avg_loss_v={}, perplexity={}".format(avg, math.exp(avg)))
	return avg

default_params = {
	'clip_norm': 5.0,
	'learning_rate': 1.0,
	'hidden_size': 650,
	'epochs': 39,
	'win_size': 35,
	'epoch_decay_start': 6,
	'decay': 1.2,
	'dropout': 0.5,
	'init_scale': 0.05
}

train = load_data('data/ptb.train.txt')
valid = load_data('data/ptb.valid.txt')
test = load_data('data/ptb.test.txt')

#import pdb; pdb.set_trace()
train=torch.LongTensor(train.astype(np.int64))
valid=torch.LongTensor(valid.astype(np.int64))
test=torch.LongTensor(test.astype(np.int64))

print(train.shape)
print(len(vocab))

model = create_model(default_params)
model.init_weights()

criterion=torch.nn.CrossEntropyLoss(size_average=False)
train_model(model, train, criterion, default_params, valid, test)
test_model(model, train, criterion, default_params)
test_model(model, valid, criterion, default_params)
test_model(model, test, criterion, default_params)

