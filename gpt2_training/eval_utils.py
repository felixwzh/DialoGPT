#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import torch
import logging

import numpy as np
import math
from pycocoevalcap.bleu.bleu import Bleu
from collections import defaultdict

logger = logging.getLogger(__name__)

EOS_ID = 50256


def cal_BLEU_4(generated, reference, is_corpus=False):
    BLEUscore = [0.0, 0.0, 0.0, 0.0]
    for idx, g in enumerate(generated):
        if is_corpus:
            score, scores = Bleu(4).compute_score(reference, {0: [g]})
        else:
            score, scores = Bleu(4).compute_score({0: [reference[0][idx]]},
                                                  {0: [g]})
        for i, s in zip([0, 1, 2, 3], score):
            BLEUscore[i] += s
    BLEUscore[0] = BLEUscore[0]/len(generated)
    BLEUscore[1] = BLEUscore[1]/len(generated)
    BLEUscore[2] = BLEUscore[2]/len(generated)
    BLEUscore[3] = BLEUscore[3]/len(generated)
    return BLEUscore


def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score


def eval_model_loss(model, tokenizer, eval_dataloader, epoch_id, args):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, '
                'please change it back to train after calling this function')
    model.eval()
    tot_loss = []
    tot_ppl = []
    tot_sample = []
    loss_total=0
    label_total=0
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, src_len, tgt_len, persona_ids = batch
            """ 
            # FIXME: may use it later
            if args.no_token_id:
                token_ids = None
            """
            n_sample = input_ids.shape[0]
            loss, ppl, loss_sum,label_size = model(input_ids, persona_ids, position_ids, token_ids, label_ids)
            tot_loss.append(loss.mean().item() * n_sample)
            tot_ppl.append(ppl.mean().item() * n_sample)
            tot_sample.append(n_sample)
            loss_total+=loss_sum
            label_total+=label_size
    # print('='*40)
    # print('eval data')
    # print('input_ids')
    # print(input_ids)
    # print('position_ids')
    # print(position_ids)
    # print('token_ids')
    # print(token_ids)
    # print('label_ids')
    # print(label_ids)
    # print('loss_total')
    # print(loss_total)
    # print('label_total')
    # print(label_total)
    print(f"\n Epoch {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} Val ppl {np.sum(tot_ppl) / np.sum(tot_sample)} ")
    print(f"\n Epoch {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} Val ppl {math.exp(loss_total.cpu().item()/label_total.cpu().item())} ")
    return np.sum(tot_loss) / np.sum(tot_sample), np.sum(tot_ppl) / np.sum(tot_sample)



def inference_model_results(model, tokenizer, inference_dataloader, args):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, '
                'please change it back to train after calling this function')
    model.eval()
    tot_sample = []
    with torch.no_grad():
        for step, batch in enumerate(inference_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, src_len, tgt_len, persona_ids = batch
            # if args.no_token_id:
            #     token_ids = None
            n_sample = input_ids.shape[0]
            logits = model.inference(input_ids, persona_ids, position_ids, token_ids)
            def decode(batch_data, tokenizer, input_flag):
                results = []
                batch_data = batch_data.cpu().data.numpy()
                for one_logits in batch_data:  # [sentence_len, vocabulary_size]
                    if not input_flag:
                        word_ids = np.argmax(one_logits, axis=1)
                    else:
                        word_ids = one_logits
                    words = []
                    for id in word_ids:
                        if tokenizer.decoder[id] != "<|endoftext|>":
                            words.append(tokenizer.decoder[id])
                        else:
                            break
                    output_words = []
                    for word in words:
                        output_words.append(word[1:]) if word.startswith("Ġ") else output_words.append(word)
                    results.append(" ".join(output_words))
                return results

            posts = decode(input_ids, tokenizer, True)
            inferences = decode(logits, tokenizer, False)

            tot_sample.append(n_sample)
            logger.info("model inference results")
            for index in range(len(posts)):
                print("post: ", posts[index])
                print("inference: ", inferences[index])

            # print(inferences)
            break
    # todo
    return None