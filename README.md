**Feel Free to create an Issue and chat with me on the Issue -- I will reply in at most 48 hours.**

This repo contains all the code(preprocessing, postprocessing, modeling, training) used in the [BioNLP paper](https://arxiv.org/abs/2305.19120) "Comparing and combining some popular NER approaches on Biomedical tasks".  
The code for all models is in `./models`.  
Utility code is in `./utils`.   
Code for preprocessing datasets (getting data ready for training) can be found in `./preprocessing`  
Configurations for all experiments, models, and datasets are in `./configs`.   

## Table of Contents  
- [Requirements](#requirements)  
- [Preprocessed Data](#preprocess)
- [Training](#training)
- [Training the three models](#three)
- [Training the Meta model](#meta)
- [Preprocessing](#preprocessing)
  - [Preprocessing data for NER models](#preprocessing-ner)
  - [Preprocessing data for Meta](#preprocessing-meta)


<a name="requirements"/>

## Requirements
Works with `Python version 3.10` and above ; We use an [Anaconda](https://www.anaconda.com/download) python 3.10 environment.
You can create an anaconda python 3.10 environment called `my_env` with:
```bash
conda create --name my_env python=3.10
```

- [allennlp](https://github.com/allenai/allennlp) : Install with:
```bash
# On Debian and Ubuntu
sudo apt-get -y install pkg-config
sudo apt-get -y install build-essential

pip install allennlp
```

All remaining dependencies can be installed by running `./setup_env.sh`, which simply includes:
```bash
echo "Install Requirements"
pip install spacy
pip install pudb
pip install colorama
pip install gatenlp
pip install pandas
pip install transformers
pip install flair
pip install benepar
pip install ipython
pip install overrides

echo "Installing Pytorch"
pip install torch torchvision torchaudio --upgrade
```

<a name="preprocess"/>

## Preprocessed data
Run script `./download_preprocessed_data.sh` to download all preprocessed training data into `./preprocessed_data`. 


<a name="training"/>

## Training
`train.py` is the main script for running experiments -- experiments involve training 
and evaluating models. Running `python train.py` (no arguments) will start a test-run, 
which will train the models on a very small subset of the data and then evaluate them 
on the validation set. The models' performance results are stored in `./training_results/performance`.


<a name="three"/>

### Training and evaluating `SEQ`, `SpanPred`, and `SeqCRF` models.
To train `SEQ`, `SpanPred`, and `SeqCRF` models on SocialDisNER and evaluate on test data. Run:
```
python train.py --production --test --experiment=social_dis_ner_experiment
```
`--production`: indicates that *all* the training data should be used.  
`--test`: indicates that the test data should also be used to evaluate the models(other than the validation data).  
The details of the experiment can be found in `./configs/experiment_configs/social_dis_ner_experiment.py`  
<br/><br/> 
Similarly, for training the three models on GENIA, run :
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

<a name="meta"/>

### Training and evaluating the `Meta` model.
For SocialDisNER, run:
```
python train.py --production --test --experiment=social_dis_ner_meta
```
For GENIA, run:
```
python train.py --production --test --experiment=genia_meta
```
For NCBI Disease, run:
```
python train.py --production --test --experiment=ncbi_disease_meta
```
For LivingNER, run:
```
python train.py --production --test --experiment=living_ner_meta
```

<a name="preprocessing"/>

## Preprocessing
Preprocessing involves preparing training data for models.

<a name="preprocessing-ner"/>

### Preprocessing data for NER models
Use the available prepreocessing configurations for NER models in `all_preprocessing_configs.py` to prepare training data for the 3 NER models.  
For example, `./preprocess.py` shows how to preprocess the Genia dataset.

<a name="preprocessing-meta"/>

### Preprocessing data for Meta
Use the Meta preprocessing configurations in `all_preprocessing_configs.py` to prepare data for Meta.
For example, `./preprocess_meta.py` uses the predictions made by `SEQ` and `SpanPred` on GENIA in `./raw_data_for_meta/genia` to prepare data for Meta. 
