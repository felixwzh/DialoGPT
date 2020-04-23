import os
import torch
import torch.nn.functional as F
import codecs


from lsp_model import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Adam
from gpt2_training.train_utils import load_model, boolean_string, set_lr, get_eval_list_same_length


import argparse
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
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


def recalc(conditioned_tokens,generated_tokens,src_pid,args,src_len,present):
	# global conditioned_tokens
	# global generated_tokens
	# for segment display purpose, keep 2 sets of tokens


		
	indexed_tokens = conditioned_tokens + generated_tokens
	position_ids=list(range(len(indexed_tokens)))

	if len(generated_tokens)>0:
		token_type_ids = [0]*(src_len-1)+[1]*(len(indexed_tokens)-src_len+1)
		assert len(token_type_ids)==len(indexed_tokens)
	else:
		token_type_ids = [0]*(src_len-1)+[1]
		assert len(token_type_ids)==len(indexed_tokens)


	tokens_tensor = torch.tensor([indexed_tokens])
	tokens_tensor = tokens_tensor.to('cuda')
	persona_ids = torch.tensor([src_pid]).to('cuda')
	position_ids = torch.tensor(position_ids).to('cuda')
	token_type_ids = torch.tensor(token_type_ids).to('cuda')

	with torch.no_grad():
		outputs = model(input_ids=tokens_tensor,
						persona_ids=persona_ids,
						position_ids=position_ids,
						token_type_ids=token_type_ids,
						lm_labels=None,
						past=None)
						# past=present) # TODO: figure out what is happening in this past.
		lm_logits, present = outputs
	logits = lm_logits[0, -1, :]
	filtered_logits = top_p_filtering(logits)
	probabilities = F.softmax(filtered_logits, dim=-1)
	next_token = torch.multinomial(probabilities, 1)
	generated_tokens.append(next_token.item())
	return next_token.item(),conditioned_tokens,generated_tokens,present

def generate(conditioned_tokens,generated_tokens,src_pid,args,src_len):
	# global conditioned_tokens
	# global generated_tokens
	present=None
	while True:
		result,conditioned_tokens,generated_tokens,present = recalc(conditioned_tokens,generated_tokens,src_pid,args,src_len,present)
		if result == 50256 or len(generated_tokens)>256:
      	# end-of-text : 50256
      	# use this special token to split segments
			return tokenizer.decode(generated_tokens[:-1])

def generate_one_sent(input_sent,src_pid,args):
	conditioned_tokens = []
	generated_tokens = []
	conditioned_tokens=reinput(input_sent)
	src_len=len(conditioned_tokens)
	output_sent=generate(conditioned_tokens,generated_tokens,src_pid,args,src_len)
	return output_sent

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', type=str, default='data/testing',
					help='the folder that contains your dataset and vocabulary file')
parser.add_argument('--decode_file', type=str, default='test.txt')
parser.add_argument('--model_folder', type=str, default='save/testing')
parser.add_argument('--model_name', type=str, default='model')
parser.add_argument('--output_folder', type=str, default='outputs')
parser.add_argument('--output_file', type=str, default='output.txt')
parser.add_argument('--decode_num', type=int, default=-1)

parser.add_argument('--decode_start', type=int, default=-1)
parser.add_argument('--decode_end', type=int, default=-1)

#####
parser.add_argument('--model_name_or_path', type=str,
                    help='pretrained model name or path to local checkpoint')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=128)

parser.add_argument("--skip_eval", action='store_true',
                    help='If true, skip evaluation.')
parser.add_argument("--init_checkpoint", type=str)
# parser.add_argument("--train_input_file", type=str)
# parser.add_argument("--eval_input_file", type=str)
parser.add_argument("--continue_from", type=int, default=0)

parser.add_argument("--train_batch_size", type=int, default=4,
                    help="batch size now means per GPU per step")
parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                    help="to increase effective batch size "
                         "and reduce synchronization")
parser.add_argument("--eval_batch_size", type=int, default=4)
# FIXME: looks like the eval_batch_size will affect the ppl score greatly. but why?
# parser.add_argument("--learning_rate", type=float, default=1e-5)
# parser.add_argument("--num_optim_steps", type=int, default=1000000,
#                     help="new API specifies num update steps")
# parser.add_argument("--valid_step", type=int, default=10000,
#                     help="how many optim steps between validations")
# parser.add_argument("--warmup_proportion", type=float, default=0.1)
# parser.add_argument("--warmup_steps", type=int, default=16000)
parser.add_argument("--normalize_data", type=boolean_string, default=True)
parser.add_argument("--fp16", type=boolean_string, default=True)
# parser.add_argument("--lr_schedule", type=str,
                    # choices=['noam', 'noamwd', 'BERT', 'None'], default='noam')
# parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument("--no_token_id", type=boolean_string, default=True) # FIXME: should we use token_id or not?

# parser.add_argument("--output_dir", type=str)
# parser.add_argument("--log_dir", type=str)
# parser.add_argument('--pbar', type=boolean_string, default=True, help='turn on progress bar')

# distributed
# parser.add_argument('--local_rank', type=int, default=-1,
                    # help='for torch.distributed')
parser.add_argument('--config', help='JSON config file')

# speaker 
parser.add_argument("--persona_emb_type", type=str,default='decode',
                    help="[decode|all｜none], `decode`: only add persona_emb to the decode part"
                         "`all`: add persona_emb to all the positions"
                         "`none`: add no persona_emb, used for baseline")
parser.add_argument("--PersonaNum", type=int, default=4167,help='number of persona')

#####

args = parser.parse_args()
if __name__ == '__main__':
	# random seed
	np.random.seed(args.seed)
	torch.random.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)

	decode_file=os.path.join(args.data_folder,args.decode_file)
	model_file=os.path.join(args.model_folder,args.model_name)
	output_file=os.path.join(args.output_folder,'{}_{}'.format(args.decode_file,args.output_file))
	Path(args.output_folder).mkdir(parents=True, exist_ok=True)

	with codecs.open(output_file, 'w', encoding='utf-8') as fout:
		fout.write("")
	# load
	tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)


	# load 
	config = GPT2Config.from_json_file(
    		os.path.join(args.model_name_or_path, 'config.json'))
	config.no_token_id=args.no_token_id
	config.persona_emb_type=args.persona_emb_type
	config.PersonaNum=args.PersonaNum
	args.n_gpu = 1
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.device=device


	model = load_model(GPT2LMHeadModel(config), model_file,
                   args, verbose=True)

	# fix misused key value
	model.eval()
	model.to('cuda')


	# decode_size = len(open(decode_file,'rU').readlines())
	output_lines=[]
	# with open(decode_file,'r') as fin:
	with codecs.open(decode_file,'r', encoding='utf-8') as fin:
		print(decode_file)
		lines=fin.readlines()
		assert args.decode_num<=len(lines)
		if args.decode_num==-1:
			decode_size=len(lines)
		else:
			decode_size=args.decode_num
		decode_list=list(range(decode_size))
		if args.decode_start!=-1:
			decode_size=args.decode_end - args.decode_start
			decode_list=list(range(args.decode_start,args.decode_end,1))
		progress = tqdm(unit_scale=True, total=decode_size,  desc="Decoding {}".format(args.decode_file))
		for i in decode_list:
			line=lines[i]
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
			
			src_pid=int(src_tokens[0])
			tgt_sent=tgt
			output_sent=generate_one_sent(src_sent,src_pid,args)
			output_line=src+'￨'+str(src_pid)+' '+output_sent+'\n'
			output_lines.append(output_line)

	
			# with open(output_file,'a') as fout:
			with codecs.open(output_file, 'a', encoding='utf-8') as fout:
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
