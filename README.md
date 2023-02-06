# CURE: A Pre-training Framework on Large-scale Patient Data for Treatment Effect Estimation

## Requirements
Install PyTorch==1.17.1 by following the instructions from [the official website](https://pytorch.org).

Install the remaining dependencies by running the script,
```
pip install -r requirements.txt
```

## Pre-train CURE

```
python model/pretrain.py --data_path data/preprocessed_data/cad/cohort --vocab_file data/preprocessed_data/cad/vocab.txt --do_train --max_steps 100000 --learning_rate 1e-4 --overwrite_output_dir --output_dir output/all_masked_prediction_cad_bertbase --mask_prediction --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --validation_split_percentage 1 --logging_steps 100 --save_steps 5000 --max_seq_length 256 --baseline_window 360 --cache_dir cache/ --time_embedding --warmup_steps 10000
```

## Fine-tune CURE for treatment effect estimation

```
python model/finetune_TEE.py --model_name_or_path output/all_masked_prediction_cad_bertbase --data_path data/preprocessed_data/cad/cohort --target_drug Rivaroxaban.json --compared_drug Aspirin.json --vocab_file data/preprocessed_data/cad/vocab.txt --do_train --do_eval --num_train_epoch 2 --learning_rate 5e-5 --overwrite_output_dir --output_dir output/all_masked_prediction_cad_bertbase_finetuned_outcome_prediction_cad --outcome_prediction --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --validation_split_percentage 10 --cache_dir cache/ --logging_steps 50 --save_steps 10000 --max_seq_length 256 --baseline_window 360 --overwrite_cache --time_embedding 
```