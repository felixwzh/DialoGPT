import os
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import argparse
from tqdm.auto import tqdm
from pathlib import Path

def reinput(text):
	# global conditioned_tokens
	# os.system('cls' if os.name == 'nt' else 'clear')
	conditioned_tokens = tokenizer.encode(text) + [50256]
	return conditioned_tokens


def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
  """
  Credit: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
  """
  assert logits.dim() == 1  # batch size 1 for single word generation
  sorted_logits, sorted_indices = torch.sort(logits, descending=True)
  cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
  # remove tokens with cumulative probability above the threshold
  sorted_indices_to_remove = cumulative_probs > top_p
  # shift the indices to the right to keep also the first token above the threshold
  sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
  sorted_indices_to_remove[..., 0] = 0
  indices_to_remove = sorted_indices[sorted_indices_to_remove]
  logits[indices_to_remove] = filter_value
  return logits


def recalc(conditioned_tokens,generated_tokens):
	# global conditioned_tokens
	# global generated_tokens
  # for segment display purpose, keep 2 sets of tokens
	indexed_tokens = conditioned_tokens + generated_tokens
	tokens_tensor = torch.tensor([indexed_tokens])
	tokens_tensor = tokens_tensor.to('cuda')
	with torch.no_grad():
	    outputs = model(tokens_tensor)
	    predictions = outputs[0]
	logits = predictions[0, -1, :]
	filtered_logits = top_p_filtering(logits)
	probabilities = F.softmax(filtered_logits, dim=-1)
	next_token = torch.multinomial(probabilities, 1)
	generated_tokens.append(next_token.item())
	return next_token.item(),conditioned_tokens,generated_tokens

def generate(conditioned_tokens,generated_tokens):
	# global conditioned_tokens
	# global generated_tokens
	while True:
		result,conditioned_tokens,generated_tokens = recalc(conditioned_tokens,generated_tokens)
		if result == 50256 or len(generated_tokens)>256:
      	# end-of-text : 50256
      	# use this special token to split segments
			return tokenizer.decode(generated_tokens[:-1])

def generate_one_sent(input_sent):
	conditioned_tokens = []
	generated_tokens = []
	conditioned_tokens=reinput(input_sent)
	output_sent=generate(conditioned_tokens,generated_tokens)
	return output_sent

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', type=str, default='data/testing',
					help='the folder that contains your dataset and vocabulary file')
parser.add_argument('--decode_file', type=str, default='test.txt')
parser.add_argument('--model_folder', type=str, default='save/testing')
parser.add_argument('--model_name', type=str, default='model')
parser.add_argument('--output_folder', type=str, default='outputs')
parser.add_argument('--output_file', type=str, default='output.txt')

args = parser.parse_args()
if __name__ == '__main__':
	decode_file=os.path.join(args.data_folder,args.decode_file)
	model_file=os.path.join(args.model_folder,args.model_name)
	output_file=os.path.join(args.output_folder,'{}_{}'.format(args.decode_file,args.output_file))
	Path(args.output_folder).mkdir(parents=True, exist_ok=True)

	with open(output_file,'w') as fout:
		fout.write("")
	# load
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	weights = torch.load(model_file)
	medium_config = GPT2Config(n_embd=1024,n_layer=24,n_head=16)
	model = GPT2LMHeadModel(medium_config)

	# fix misused key value
	weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
	weights.pop("lm_head.decoder.weight",None)

	model.load_state_dict(weights)
	model.eval()
	model.to('cuda')


	# decode_size = len(open(decode_file,'rU').readlines())
	output_lines=[]
	with open(decode_file,'r') as fin:
		lines=fin.readlines()

		progress = tqdm(unit_scale=True, total=len(lines),  desc="Decoding {}".format(args.decode_file))
		for line in lines:
			progress.update(1)
		# for i in tqdm.tqdm(range(len(lines))):
		# 	line=lines[i]
		# for line in tqdm.tqdm(lines):
			# 474 what are those weird lines one sees after rubbing their eyes ?￨474 dream dust
			src,tgt=line.strip().split('￨')
			src_tokens=src.split(' ')
			src_sent=''
			for word in src_tokens[1:]:
				src_sent+=word
				src_sent+=' '
			src_sent=src_sent[:-1]
			src_pid=src_tokens[0]
			tgt_sent=tgt
			output_sent=generate_one_sent(src_sent)
			output_line=src+'￨'+src_pid+' '+output_sent+'\n'
			output_lines.append(output_line)

	
			with open(output_file,'a') as fout:
				fout.write(output_line)
					

	# input_sent='what are some crazy animal facts that no one knows ?'
	# output_sent=generate_one_sent(input_sent)
	# print(output_sent)

	# """
	# CUDA_VISIBLE_DEVICES=0 python gpt_decoder.py --model_folder '../models/medium_10epochs_trial_2/GPT2.1e-05.20.1gpu.2020-04-15200256' \
	# --model_name GP2-pretrain-step-12500.pkl \
	# --data_folder ../data/src_data_full_feat_tf_resplited_review \
	# --decode_file val_full_ref.txt \
	# --output_folder ../outputs/medium_10epochs_trial_2 \
	# --output_file step-12500.txt
	# """
