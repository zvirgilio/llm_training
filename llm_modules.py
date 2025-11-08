from dataclasses import dataclass 
import torch
import torch.nn as nn
from torch.nn import functional as F 

#----------------------------------------------------------------------------------------------------

@dataclass 
class ModelConfig:
	block_size: int = 256
	vocab_size: int = 65
	n_layer: int = 6
	n_head: int = 6
	n_embed: int = 384

#----------------------------------------------------------------------------------------------------

class TransformerBlock(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.ln_1 = nn.LayerNorm(config.n_embed)
		self.attn = CausalSelfAttention(config)
		self.ln_2 = nn.LayerNorm(config.n_embed)
		self.mlp = MLP(config)


	def forward(self, x):
		''' x + attention + mlp( x + attention )
			keeps a residual stream and forks off the learned block
			'''
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))

		return x

#----------------------------------------------------------------------------------------------
class MLP(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.c_fc 	= nn.Linear(config.n_embed, 4 * config.n_embed)
		self.gelu 	= nn.GELU()	# gelu fixes 0 gradient component of ReLU, no approximate since doesn't need speed up nowadays
		self.c_proj	= nn.Linear(config.n_embed * 4, config.n_embed)


	def forward(self, x):
		x = self.c_fc(x)
		x = self.gelu(x)
		x = self.c_proj(x)
		return x

#----------------------------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):

	def __init__(self, config):
		super().__init__()

		assert config.n_embed % config.n_head == 0

		# key, query, value projections
		self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)

		# output projection
		self.c_proj = nn.Linear(config.n_embed, config.n_embed)

		# regularization
		self.n_head = config.n_head
		self.n_embed = config.n_embed

		# maks more than buffer
		self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
										  .view(1, 1, config.block_size, config.block_size))


	def forward(self, x):
		B, T, C = x.size() #batch size, sequence length (block_size), embedding dimension (n_embed)

		# calculate query, key and values for all heads in a batch and move head forward to be the batch
		# nh: "number of heads", hs: "head size" and C: "number of channels" = nh * ns
		qkv = self.c_attn(x)
		q, k, v = qkv.split(self.n_embed, dim=2)

		k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)		# B x nh x T x hs
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)		# B x nh x T x hs
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)		# B x nh x T x hs

		att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))		# (B x nh x T x hs) @ (B x nh x hs x T) = B x nh x T x T
		att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
		att = F.softmax(att, dim = -1)
		y = att @ v 														# (B x nh x T x T) @ (B x nh x T x hs) = B x nh x T x hs
		y = y.transpose(1, 2).contiguous().view(B, T, C)					# re-assemble all head outputs side by side

		# output projection
		y = self.c_proj(y)
		return y 

#----------------------------------------------------------------------------------------------