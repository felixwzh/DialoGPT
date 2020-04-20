CUDA_VISIBLE_DEVICES=0 python gpt_decoder.py --model_folder '../../DialoGPT/models/medium_10epochs_trial_4/GPT2.1e-05.20.1gpu.2020-04-18015154' \
                --model_name GP2-pretrain-step-250000.pkl \
                --data_folder ../data/src_data_full_feat_tf_resplited_review \
                --decode_file test_ref.txt --decode_num -1 \
                --output_folder ../outputs/medium_10epochs_trial_4 \
                --output_file step-250000.txt
                
CUDA_VISIBLE_DEVICES=0 python gpt_decoder.py --model_folder '../../DialoGPT/models/medium_10epochs_trial_4/GPT2.1e-05.20.1gpu.2020-04-18015154' \
                --model_name GP2-pretrain-step-250000.pkl \
                --data_folder ../data/src_data_full_feat_tf_resplited_review \
                --decode_file val_ref.txt --decode_num -1 \
                --output_folder ../outputs/medium_10epochs_trial_4 \
                --output_file step-250000.txt
                
CUDA_VISIBLE_DEVICES=0 python gpt_decoder.py --model_folder '../../DialoGPT/models/medium_10epochs_trial_4/GPT2.1e-05.20.1gpu.2020-04-18015154' \
                --model_name GP2-pretrain-step-250000.pkl \
                --data_folder ../data/src_data_full_feat_tf_resplited_review \
                --decode_file train_ref.txt --decode_num 10000 \
                --output_folder ../outputs/medium_10epochs_trial_4 \
                --output_file step-250000.txt
                
