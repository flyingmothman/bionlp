**Feel Free to create an Issue and Chat with me in the Issues section.** 

This repo **will soon** contain all the code(preprocessing, postprocessing, modeling, training) used in the paper "Comparing and combining some popular NER approaches on Biomedical tasks".  
The code for all models is in `./models`.  
Utility code is in `./utils`.  
Configurations for all experiments, models, and datasets is in `./configs`.  

## Requirements
Works with `Python version 3.10` and above ; We use an Anaconda python 3.10 environment and install dependencies using `pip install -r requirements.txt`.
You can create an anaconda python 3.10 environment called `my_env` with:
```
conda create -n my_env python=3.10
```
- [fzf](https://github.com/junegunn/fzf) : Install with
```
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
yes | ~/.fzf/install
```
- [allennlp](https://github.com/allenai/allennlp) : Install with
```
# on Debian and Ubuntu
sudo apt-get -y install pkg-config
sudo apt-get -y install build-essential
pip install allennlp
```

## Preprocessed data
Run script `./download_preprocessed_data.sh` to download all preprocessed training data into `./preprocessed_data`. The models' performance results are persisted at `./training_results/performance`.

## Training
`train.py` is the main script for running experiments -- experiments involve training and evaluating models.  
Running `python train.py` (no arguments) will start a test-run, which will train the 
models on a very small subset of the data and then evaluate them on the validation set.  

To train `SEQ`, `SpanPred`, and `SeqCRF` models on SocialDisNER and evaluate on test data, run:
```
python train.py --production --test
```
`--production`: indicates that *all* the training data should be used.  
`--test`: indicates that the test data should also be used to evaluate the models(other than the validation data).  

As soon as it starts, `train.py` will prompt the user to select an experiment.  
Select `social_dis_ner_experiment` from the menu.  
