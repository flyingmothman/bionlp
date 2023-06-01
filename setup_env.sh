echo "Install requirements"
pip install spacy
pip install pudb
pip install colorama
pip install gatenlp
pip install pandas
pip install transformers
pip install allennlp
pip install flair
pip install benepar
pip install ipython
pip install overrides

echo "Installing Pytorch"
# conda install -y pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install torch torchvision torchaudio --upgrade
