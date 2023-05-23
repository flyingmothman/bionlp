**Feel Free to create an Issue and Chat with me in the Issues section.** 

This repo **will soon** contain all the code(preprocessing, postprocessing, modeling, training) used in the paper "Comparing and combining some popular NER approaches on Biomedical tasks".  
Works with `Python version 3.10` and above ; We use an Anaconda python 3.10 environment and install dependencies using `pip install -r requirements.txt`.  
The code for all models is in `./models`.  
Utility code is in `./utils`.  
Configurations for all experiments, models, and datasets is in `./configs`.  

## Preprocessed data
Run script `./download_preprocessed_data.sh` to download all preprocessed data into `./preprocessed_data`.  

## Training
`train.py` is the main script for running experiments.    
`python train.py` will do a dry run on the selected experiment.  
To train `SEQ`, `SpanPred`, and `SeqCRF` models on SocialDisNER and evaluate on test data, run:
```
python train.py --production --test
```
Then select `social_dis_ner_experiment` from the menu.  
The models' performance results are persisted at `./training_results/performance`.  
