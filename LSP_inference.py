#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
'''
 * @Desc: train GPT2 from scratch/ fine tuning.
          Modified based on Huggingface GPT-2 implementation
'''

import json
import os
import sys
import argparse
import logging
import time
import tqdm
import datetime
import torch
from pathlib import Path
import numpy as np

from os.path import join
from torch.distributed import get_rank, get_world_size

from lsp_model import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Adam
from gpt2_training.train_utils import load_model, boolean_string, set_lr, get_eval_list_same_length
from gpt2_training.eval_utils import eval_model_loss, inference_model_results

from data_loader import BucketingDataLoader, DynamicBatchingLoader, DistributedBucketingDataLoader

from gpt2_training.distributed import all_reduce_and_rescale_tensors, all_gather_list

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

INF = 100000000
CACHE_EMPTY_STEP = 10000
EVAL_STEP = 10000

#########################################################################
# Prepare Parser
##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True,
                    help='pretrained model name or path to local checkpoint')

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=128)

parser.add_argument("--init_checkpoint", type=str, required=True)
parser.add_argument("--inference_input_file", type=str, required=True)

parser.add_argument("--inference_batch_size", type=int, default=8)
parser.add_argument("--num_optim_steps", type=int, default=1000000,
                    help="new API specifies num update steps")

parser.add_argument("--fp16", type=boolean_string, default=True)
parser.add_argument("--normalize_data", type=boolean_string, default=True)
parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument("--no_token_id", type=boolean_string, default=True)

parser.add_argument("--log_dir", type=str, required=True)
# distributed
parser.add_argument('--local_rank', type=int, default=-1,
                    help='for torch.distributed')
parser.add_argument('--config', help='JSON config file')
# speaker 
parser.add_argument("--persona_emb_type", type=str,default='decode',
                    help="[decode|all], `decode`: only add persona_emb to the decode part"
                         "`all`: add persona_emb to all the positions")
parser.add_argument("--PersonaNum", type=int, default=4167,help='number of persona')

# do normal parsing
args = parser.parse_args()

if args.config is not None:
    # override argparse defaults by config JSON
    opts = json.load(open(args.config))
    for k, v in opts.items():
        if isinstance(v, str):
            # PHILLY ENV special cases
            if 'PHILLY_JOB_DIRECTORY' in v:
                v = v.replace('PHILLY_JOB_DIRECTORY',
                              os.environ['PHILLY_JOB_DIRECTORY'])
            elif 'PHILLY_LOG_DIRECTORY' in v:
                v = v.replace('PHILLY_LOG_DIRECTORY',
                              os.environ['PHILLY_LOG_DIRECTORY'])
        setattr(args, k, v)

    # command line should override config JSON
    argv = sys.argv[1:]
    overrides, _ = parser.parse_known_args(argv)
    for k, v in vars(overrides).items():
        if f'--{k}' in argv:
            setattr(args, k, v)
    setattr(args, 'local_rank', overrides.local_rank)

if args.local_rank == -1:
    logger.info('CUDA available? {}'.format(str(torch.cuda.is_available())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu
else:
    # distributed training
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    # Initializes the distributed backend which will take care of
    # sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
    n_gpu = torch.distributed.get_world_size()
    args.device, args.n_gpu = device, 1
    logger.info("device: {} n_gpu: {}, distributed training: {}, "
                "16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
log_dir = args.log_dir
Path(log_dir).mkdir(parents=True, exist_ok=True)

logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

#########################################################################
# Prepare Data Set
##########################################################################
print("Prepare Data")
enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

config = GPT2Config.from_json_file(
    join(args.model_name_or_path, 'config.json'))

inference_dataloader_loss = DynamicBatchingLoader(
    args.inference_input_file, enc, args.normalize_data,
    args.inference_batch_size, args.max_seq_length, True)

inference_dataloader_gen = get_eval_list_same_length(
    args.inference_input_file, enc, args.inference_batch_size, True)

# eval_dataloader_loss = DynamicBatchingLoader(
#     args.eval_input_file, enc, args.normalize_data,
#     args.eval_batch_size, args.max_seq_length)
#
# eval_dataloader_gen = get_eval_list_same_length(
#     args.eval_input_file, enc, args.eval_batch_size, True)

#########################################################################
# Prepare Model
##########################################################################
# add args to config 
config.no_token_id=args.no_token_id
config.persona_emb_type=args.persona_emb_type
config.PersonaNum=args.PersonaNum

print("Prepare Model")
logger.info("Prepare Model")
model = load_model(GPT2LMHeadModel(config), args.init_checkpoint,
                   args, verbose=True)

if args.local_rank != -1:
    # when from scratch make sure initial models are the same
    params = [p.data for p in model.parameters()]
    all_reduce_and_rescale_tensors(params, float(torch.distributed.get_world_size()))

no_decay = ['bias', 'ln']  # no decay for bias and LayerNorm (ln)

#########################################################################
# Inference !
##########################################################################
print("Model inference")
logger.info("Model inference")

inference_logger = open(join(log_dir, 'inference_log.txt'), 'a+', buffering=1)

epoch = 0

if args.local_rank != -1:
    n_gpu = 1
# todo modify loss out.
results = inference_model_results(model, enc, inference_dataloader_loss, args)
# todo output format
# print('{},{},{},{},{}'.format(epoch + 1, global_step + 1, step + 1, eval_loss, eval_ppl), file=inference_logger)
logger.info("inference_final_results:")
if results is None:
    logger.info("current results are None")
else:
    logger.info(results)

inference_logger.close()
