import torch
import torch.nn as nn
from torch.nn import functional as F 

from llm_modules import ModelConfig, TransformerBlock



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


	# use this method to load the weights from the pre-trained gpt2 from huggingface
	@classmethod
	def from_pretrained(cls, model_type):
		assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

		from transformers import GPT2LMHeadModel
		print("loading weights from pretrained gpt: %s" % model_type)

		# n_layer, n_head and n_embed are determined by which model we are loading
		config_args = {
			'gpt2':			dict(n_layer=12, n_head=12, n_embed=768),
			'gpt2-medium':	dict(n_layer=24, n_head=16, n_embed=1024),
			'gpt2-large':	dict(n_layer=36, n_head=20, n_embed=1280),
			'gpt2-xl':		dict(n_layer=48, n_head=25, n_embed=1600),
		}[model_type]

		config_args['vocab_size'] = 50257 # always 50257 for gpt model checkpoints
		config_args['block_size'] = 1024  # alwats 1024 for gpt model checkpoints

		# create a from-scratch initialized minGPT model
		config = ModelConfig(**config_args) 
		model = LLMModel(config)

		sd = model.state_dict()
		sd_keys = sd.keys()
		sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard the mask/buffer

		# init a huggerface/transformers model
		print('Loading HF model weights')
		model_hf = GPT2LMHeadModel.from_pretrained(model_type)
		sd_hf = model_hf.state_dict()


		# copy the weights
		# make sure all parameters are aligned

		sd_keys_hf = sd_hf.keys()
		sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore masks
		sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
		transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

		# openai checkpoints were a conv1d module, we want linear
		# so we need to transpose the weights when copying them over

		assert len(sd_keys_hf) == len(sd_keys), f"Mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

		for k in sd_keys_hf:
			if any(k.endswith(w) for w in transposed):
				# transpose if needed
				assert sd_hf[k].shape[::-1] == sd[k].shape
				with torch.no_grad():
					sd[k].copy_(sd_hf[k].t())

			else:
				assert sd_hf[k].shape == sd[k].shape
				with torch.no_grad():
					sd[k].copy_(sd_hf[k])

		return model




def main():
	model = LLMModel.from_pretrained('gpt2')
	print("didn't crash")




if __name__ == '__main__':
	main()