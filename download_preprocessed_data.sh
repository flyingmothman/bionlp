printf "\n\n\n Downloading preprocessed data file from dropbox \n\n\n"
curl -L -o ./preprocessed_data/bionlp_preprocessed_data.zip https://www.dropbox.com/s/kvlzlnvgaafeftq/bionlp_preprocessed_data.zip?dl=0

printf "\n\n\n Unzipping downloaded data into ./preprocessed_data \n"
unzip preprocessed_data/bionlp_preprocessed_data.zip -d preprocessed_data/
