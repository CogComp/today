# Introduction
This is the data and code repository for our ACL 2023 paper ["Generic Temporal Reasoning with Differential Analysis and Explanation"](http://cogcomp.org/page/publication_view/1008).

# TODAY
TODAY is our crowdsourced dataset. The TODAY dataset and its overall framework are designed to evaluate systems’ ability to make temporal predictions with plausible reasons.
## Dataset
We include the dataset under `data/`. 

# Models and Experiments
We provide our codebase to reproduce the experiment results reported in the paper. All models can be found on [this page](http://cogcomp.org/page/model_view/9).

## Pre-trained Models
- Download the entire directory [`ptntime-pretrained-model`](http://cogcomp.org/models/today_models/ptntime-pretrained-model.zip)
and put it under `model/ptntime-pretrained-model/`. 

- Download the entire directory [`best_model_checkpoint`](http://cogcomp.org/models/today_models/best_model_checkpoint.zip)
and put it under `model/best_model_checkpoint/`. 

## Run Experiments
Work under `code/` directory (This is very important as we refer all paths relative to this working directory below).

- Install requirements by `pip install -r requirement.txt`

- To run and train the best ptntime model, use `sh train_ptntime.sh`

- To run and train the best T5 model, use `sh train_t5.sh`

- To run the baseline without TODAY, use `sh train_ptntime_wo_today.sh`

- To test the model, use `sh inference.sh`

## Generate LLM Supervision Signals
![GPT-3.5 Pipeline](pipeline.png)

### Generate GPT-3.5 Data 
- Please follow the instructions and code in `GPT_today.ipynb`.

### Train Verifiers
- We pre-trained the explanation sentence verifier. Download the entire directory [`ptntime_explanation_verifier`](http://cogcomp.org/models/today_models/ptntime_explanation_verifier.zip)
and put it under `model/ptntime_explanation_verifier/`.

- To train the general and additional sentence verifiers, use `sh train_ptntime_verifier.sh`.

### Distill GPT-3.5 Data 
- To distill the data, and further train with the distilled GPT-3.5 data, use `sh distill_ptntime.sh`.

# Citation
See the following paper: 
```
@inproceedings{feng-etal-2023-generic,
    title = "Generic Temporal Reasoning with Differential Analysis and Explanation",
    author = "Feng, Yu  and
      Zhou, Ben  and
      Wang, Haoyu  and
      Jin, Helen  and
      Roth, Dan",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.671",
    pages = "12013--12029",
   }
```
