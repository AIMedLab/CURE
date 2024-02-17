# CURE: A Pre-training Framework on Large-scale Patient Data for Treatment Effect Estimation

## Introduction
Code for paper "A Pre-training Framework on Large-scale Patient Data for Treatment Effect Estimation".

In this paper, we propose a novel transformer-based pre-training and fine-tuning framework called CURE for TEE from observational data. CURE is pre-trained on large-scale unlabeled patient data to learn representative contextual patient representations, and then fine-tuned on labeled patient data for TEE.

<img src="img/data_flow.png" title="The overall pipeline of CURE.">

We obtain and preprocess 3M large-scale observational data ([MarketScan Research Databases](www.merative.com/real-world-evidence)) and 4 downstream TEE tasks (10-20K patient samples) for evaluating the comparative treatment effectiveness for patients with coronary artery disease (CAD).

## Requirements
Install PyTorch==1.17.1 by following the instructions from [the official website](https://pytorch.org).

Install the remaining dependencies by running the script,
```
pip install -r requirements.txt
```

## Dataset
The real world patient data used in this paper is [MarketScan claims data](www.merative.com/real-world-evidence).

#### Data flow chart

The data flow chart of MarketScan claims data.
<img src="img/MarketScan_DataFlow.png" width="70%"/>

Source: 2012 MarketScan&reg; CCAE MDCR User Guide

#### Data files used 
- Inpatient Admissions (I) : Admission summary records
- Outpatient Services (O): Individual outpatient claim records
- Outpatient Pharmaceutical Claims (D): Individual outpatient prescription drug claim records
- Population (P): Summarizes demographic information about the eligible population

#### Input data demo
The demo of the input data can be found in the [data/demo](data/demo)

##### Cohort
The data structure for [cohort table](data/demo/Cohort.csv) is as follows,

| Column Name | Description                        | Note                                                                                                                                 |
|-------------|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| ENROLID     | Patient enroll ID                  | Unique identifier for each patient                                                                                                   |
| Index_date  | The date of first CAD encounter    | i.e., min (ADMDATE [1st CAD admission date for the inpatient records],SVCDATE [1st CAD service date for the outpatient records]) |
| DTSTART     | Date of insurance enrollment start | M/D/Y, e.g., 03/25/2732                                                                                                              |
| DTEND       | Date of insurance enrollment end   | M/D/Y, e.g., 03/25/2732                                                                                                              |

##### Drug table
The data structure for the [drug table](data/demo/drug/drug12.csv) is as follows,

| Column Name | Description                                                                  | Note                                                                                                                    |
|-------------|------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| ENROLID     | Patient enroll ID                                                            | Unique identifier for each patient                                                                                      |
| NDCNUM      | National drug code ([NDC](https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory))                                                     | We map NDC to observational medical<br>outcomes partnership ([OMOP](https://ohdsi.org/)) ingredient concept ID, and obtain 1,353 unique drugs |
| SVCDATE     | Date to take the prescription                                                | M/D/Y, e.g., 03/25/2732                                                                                                 |
| DAYSUPP     | Days supply. The number of days of drug therapy covered by this prescription | Day, e.g., 28                                                                                                           |

##### Inpatient table
The data structure for the [inpatient table](data/demo/inpatient/inpat12.csv) is as follows,

| Column Name  | Description                                       | Note                                                                                                                                                            |
|--------------|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ENROLID      | Patient enroll ID                                 | Unique identifier for each patient                                                                                                                              |
| DX1-DX15     | Diagnosis codes. International Classification of Diseases ([ICD](https://www.cdc.gov/nchs/icd/index.htm)) codes                                | 57,089 ICD-9/10 codes considered in the dataset. Dictionary for [ICD-9](http://www.icd9data.com/2015/Volume1/default.htm) and [ICD-10](https://www.icd10data.com/) codes.|
| DXVER        | Flag to denote ICD-9/10 codes                     | “9” = ICD-9-CM and “0” = ICD-10-CM                                                                                                                              |                                                                                                                                                            |
| ADMDATE      | Admission date for this inpatient visit           | M/D/Y, e.g., 03/25/2732                                                                                                                                         |
| Days         | The number of days stay in the inpatient hospital | Day, e.g., 28                                                                                                                                                   |

##### Outpatient table
The data structure for the [outpatient table](data/demo/outpatient/outpat12.csv) is as follows,

| Column Name  | Description                                       | Note                                                                                                                                                            |
|--------------|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ENROLID      | Patient enroll ID                                 | Unique identifier for each patient                                                                                                                              |
| DX1-DX4     | Diagnosis codes. International Classification of Diseases ([ICD](https://www.cdc.gov/nchs/icd/index.htm)) codes                                | 57,089 ICD-9/10 codes considered in the dataset. Dictionary for [ICD-9](http://www.icd9data.com/2015/Volume1/default.htm) and [ICD-10](https://www.icd10data.com/) codes.|
| DXVER        | Flag to denote ICD-9/10 codes                     | “9” = ICD-9-CM and “0” = ICD-10-CM                                                                                                                              |                                                                                                                                                             |
| SVCDATE      | Service date for this outpatient visit           | M/D/Y, e.g., 03/25/2732                                                                                                                                         |                                                                                                                                                |

##### Demographics
The data structure for [demo table](data/demo/demo.csv) is as follows,

| Column Name | Description        | Note                               |
|-------------|--------------------|------------------------------------|
| ENROLID     | Patient enroll ID  | Unique identifier for each patient |
| DOBYR       | birth year         | Year, e.g., 2099                   |
| SEX         | gender             | 1- male; 2- female                 |

## Pre-train CURE

```
python pretrain.py 
  --data_path data/preprocessed_data/cad/cohort 
  --vocab_file data/preprocessed_data/cad/vocab.txt 
  --do_train 
  --max_steps 100000 
  --learning_rate 1e-4 
  --overwrite_output_dir 
  --output_dir output/all_masked_prediction_cad_bertbase 
  --mask_prediction 
  --per_device_train_batch_size 24 
  --per_device_eval_batch_size 24 
  --validation_split_percentage 1 
  --logging_steps 100 
  --save_steps 5000 
  --max_seq_length 256 
  --baseline_window 360 
  --cache_dir cache/ 
  --time_embedding 
  --warmup_steps 10000
```

## Fine-tune CURE for treatment effect estimation

```
python finetune_TEE.py 
  --model_name_or_path output/all_masked_prediction_cad_bertbase 
  --data_path data/preprocessed_data/cad/cohort 
  --target_drug Rivaroxaban.json 
  --compared_drug Aspirin.json 
  --vocab_file data/preprocessed_data/cad/vocab.txt 
  --do_train 
  --do_eval 
  --num_train_epoch 2 
  --learning_rate 5e-5 
  --overwrite_output_dir 
  --output_dir output/all_masked_prediction_cad_bertbase_finetuned_outcome_prediction_cad 
  --outcome_prediction 
  --per_device_train_batch_size 32 
  --per_device_eval_batch_size 32 
  --validation_split_percentage 10 
  --cache_dir cache/ 
  --logging_steps 50 
  --save_steps 10000 
  --max_seq_length 256 
  --baseline_window 360 
  --overwrite_cache 
  --time_embedding 
```
