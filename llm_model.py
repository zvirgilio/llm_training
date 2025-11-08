import torch
import torch.nn as nn
from torch.nn import functional as F 

from llm_modules import ModelConfig, Block



class LLMModel(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config 

		self.transformer = nn.ModuleDict(
			dict(
				wte = nn.Embedding(config.vocab_size, config.n_embed),
				wpe = nn.Embedding(config.block_size, config.n_embed),
				h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
				ln_f = nn.LayerNorm(config.n_embed),
				)
			)
		self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
		









def main():
	gpt2Config = ModelConfig

	# set the configuration to match gp2 sizes so we can load the weights
	gpt2Config.block_size = 1024		# max sequence length
	gpt2Config.vocab_size = 50257		# number of tokens
	gpt2Config.n_layer = 1024			# number of layers
	gpt2Config.n_head = 12  			# number of heads
	gpt2Config.n_embed = 768 			# embedding dimensions




if __name__ == '__main__':
	main()