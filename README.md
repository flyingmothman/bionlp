**Feel Free to create an Issue and Chat with me in the Issues section -- I will reply in at most 48 hours.**

This repo contains all the code(preprocessing, postprocessing, modeling, training) used in the [BioNLP paper](https://arxiv.org/abs/2305.19120) "Comparing and combining some popular NER approaches on Biomedical tasks".  
The code for all models is in `./models`.  
Utility code is in `./utils`.   
Code for preprocessing datasets (getting data ready for training) can be found in `./preprocessing`  
Configurations for all experiments, models, and datasets are in `./configs`.   

## Requirements
Works with `Python version 3.10` and above ; We use an Anaconda python 3.10 environment.
You can create an anaconda python 3.10 environment called `my_env` with:
```
conda create -n my_env python=3.10
```

- [allennlp](https://github.com/allenai/allennlp) : Install with
```
# On Debian and Ubuntu
sudo apt-get -y install pkg-config
sudo apt-get -y install build-essential
pip install allennlp
```

All remaining dependencies can be installed using `pip install -r requirements.txt`. 

## Preprocessed data
Run script `./download_preprocessed_data.sh` to download all preprocessed training data into `./preprocessed_data`. 

## Training
`train.py` is the main script for running experiments -- experiments involve training 
and evaluating models. Running `python train.py` (no arguments) will start a test-run, 
which will train the models on a very small subset of the data and then evaluate them 
on the validation set. The models' performance results are persisted at `./training_results/performance`.

### Example
To train `SEQ`, `SpanPred`, and `SeqCRF` models on SocialDisNER and evaluate on test data. Run:
```
python train.py --production --test --experiment=social_dis_ner_experiment
```
`--production`: indicates that *all* the training data should be used.  
`--test`: indicates that the test data should also be used to evaluate the models(other than the validation data).  
The details of the experiment can be found in `./configs/experiment_configs/social_dis_ner_experiment.py`  
<br/><br/> 
Similarly, for training the models on GENIA, run :
```
python train.py --production --test --experiment=genia
```
For NCBI-Diease, run:
```
python train.py --production --test --experiment=ncbi
```
For LivingNER, run:
```
python train.py --production --test --experiment=living_ner
```

## Preprocessing
`preprocess.py` is the main script for preprocessing raw data.
