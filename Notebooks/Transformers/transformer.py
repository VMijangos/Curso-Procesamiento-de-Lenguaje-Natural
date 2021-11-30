from transformer_modules import *
from transformer_optimization import *
import torch
from torch import nn
from torch.autograd import Variable
import copy
from math import ceil
from tqdm import tqdm

def batch_gen(X,Y,batch):
    "Generate batch for a src-tgt task."
    num_ex = X.shape[0]
    nbatches = ceil(num_ex/batch)
    for i in range(nbatches):
        src = Variable(X[i:i+batch], requires_grad=False)
        tgt = Variable(Y[i:i+batch], requires_grad=False)
        yield Batch(src, tgt, 0)

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))                            
        prob = model.generator(out[:, -1])
        #print(prob)
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        #print(next_word)
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

class transformer_model():
	def __init__(self,src_vocab,tgt_vocab,N=1,d_model=128,d_ff=256, h=1, dropout=0.1):
		super().__init__()
		self.loss = None

		"Construye un modelo dado hiperparámetros."
		#Función para hacer copias 
		#de capas del mismo tipo
		c = copy.deepcopy
		#Atenciín con número de cabezales h y dimensión del modelo d_model
		attn = MultiHeadedAttention(h, d_model)
		#FeedForward con dimensión de capa d_ff y dropout
		ff = PositionwiseFeedForward(d_model, d_ff, dropout)
		#Positional decoding con dropout
		position = PositionalEncoding(d_model, dropout)

		#Creación del encoder y el decoder y embeddings
		encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
		decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
		src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
		tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
		generator = Generator(d_model, tgt_vocab)
		#Creación del modelo
		self.model = EncoderDecoder(encoder,decoder,src_embed,tgt_embed,generator)

		# This was important from their code. 
		# Initialize parameters with Glorot / fan_avg.
		for p in self.model.parameters():
			if p.dim() > 1:
		    		nn.init.xavier_uniform_(p) #Deprecated nn.init.xavier_uniform for nn-init.xavier_uniform_


	def train(self,X,Y,its=100,lr=0,betas=(0.9, 0.98),eps=1e-9,factor=1,warmup=4000,batch_size=1):
		X, Y = torch.tensor(X), torch.tensor(Y)
		criterion = nn.CrossEntropyLoss()
		model_opt = NoamOpt(self.model.src_embed[0].d_model, factor, warmup, torch.optim.Adam(self.model.parameters(),lr=lr, betas=betas, eps=eps))	

		loss = []
		for epoch in tqdm(range(its)):
			self.model.train()
			epoch_loss = run_epoch(batch_gen(X,Y, batch_size), self.model, SimpleLossCompute(self.model.generator, criterion, opt=model_opt))
			loss.append(epoch_loss)

		self.loss = loss

	def predict(self,x_input,max_len=5,BOS=0):
		self.model.eval()
		src = Variable(torch.LongTensor(x_input))
		n = src.shape[1]
		src_mask = Variable(torch.ones(1, 1, n) )

		return greedy_decode(self.model, src, src_mask, max_len=max_len, start_symbol=BOS).detach().reshape(max_len)
