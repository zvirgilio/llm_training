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

		#weight sharing between encoder and head layer
		self.transformer.wte.weight = self.lm_head.weight #orpahns wte.weight tensor and torch should clean it up

		# initialize params to match gpt 2
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			# std to 0.02 roughly matches 1/sqrt(N), the number of features incoming in gpt2 models (between 768 to 1600)
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, idx, targets=None):
		# idx is shape (B = batch dimension, T = sequence size)
		B, T = idx.size()
		assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

		#forward token and position embeddings
		pos = torch.arange(0, T, dtype = torch.long, device=idx.device) #shape (T)
		pos_emb = self.transformer.wpe(pos) #positions of shape (T, n_embed)
		tok_emb = self.transformer.wte(idx) #token embeddings of shape (B, T, n_embed)
		x = tok_emb + pos_emb

		# forward through the blocks of the transformer
		for block in self.transformer.h:
			x = block(x)

		#forward to the final layernorm and classifier
		x = self.transformer.ln_f(x)
		logits = self.lm_head(x) # (B, T, vocab size)

		loss = None
		if targets is not None:
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

		return logits, loss

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



def get_device():
	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"
	elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		device = "mps"
	print(f"using device: {device}")
	return device


def main():
	device = torch.device(get_device())

	num_return_sequences = 5 
	max_length = 30

	# model = LLMModel.from_pretrained('gpt2')
	model = LLMModel(ModelConfig())
	
	model.eval()
	model.to(device)

	#tokenize the prompt using tiktoken
	import tiktoken
	enc = tiktoken.get_encoding('gpt2')
	with open('input.txt', 'r') as f:
		text = f.read()
	text = text[:1000]
	tokens = enc.encode(text)
	B, T = 4, 32
	buf = torch.tensor(tokens[:B*T+1], device=device)
	x = buf[:-1].view(B, T)
	y = buf[1:].view(B, T)

	# x.to(device)
	# y.to(device)

	logits, loss = model(x, y)
	print(loss)
	import sys; sys.exit(0)

	# generate 
	torch.manual_seed(42)
	torch.mps.manual_seed(42)

	while x.size(1) < max_length:
		#forward to get logits
		with torch.no_grad():
			#run the model
			logits = model(x)

			#get the predictions for the next token
			logits = logits[:, -1, :]

			#compute probs
			probs = F.softmax(logits, dim=-1)

			# top-k sampling of 50 tokens (huggin face default)
			topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

			# select fromt he top 50 probs
			ix = torch.multinomial(topk_probs, 1)

			# gather the corresponding indices
			xcol = torch.gather(topk_indices, -1, ix)

			#append to the sequence
			x = torch.cat((x, xcol), dim=1)

	#pring the generated text
	for i in range(num_return_sequences):
		tokens = x[i, :max_length].tolist()
		decoded = enc.decode(tokens)
		print(">", decoded)

if __name__ == '__main__':
	main()