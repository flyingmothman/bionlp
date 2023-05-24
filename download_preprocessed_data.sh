printf "\n\n\n Downloading SocialDisNER data\n\n\n"

curl -L -o ./preprocessed_data/social_dis_ner_valid_default_production_samples.json https://www.dropbox.com/s/wbxbxyathl2xf7w/social_dis_ner_valid_default_production_samples.json\?dl\=0
curl -L -o ./preprocessed_data/social_dis_ner_train_default_production_samples.json https://www.dropbox.com/s/xshovt6kj1q23v0/social_dis_ner_train_default_production_samples.json?dl=0
curl -L -o ./preprocessed_data/social_dis_ner_test_default_production_samples.json https://www.dropbox.com/s/2pjbiak150zuj5e/social_dis_ner_test_default_production_samples.json?dl=0
curl -L -o ./preprocessed_data/social_dis_ner_train_vanilla_types.txt https://www.dropbox.com/s/nlvwh1rxju9c65i/social_dis_ner_train_vanilla_types.txt?dl=0



printf "\n\n\n Downloading Meta SocialDisNER data\n\n\n"

curl -L -o ./preprocessed_data/social_dis_ner_valid_config_meta_social_dis_ner_production_samples.json https://www.dropbox.com/s/d9b0nhu82ncb6yv/social_dis_ner_valid_config_meta_social_dis_ner_production_samples.json?dl=0
curl -L -o ./preprocessed_data/social_dis_ner_train_config_meta_social_dis_ner_production_samples.json https://www.dropbox.com/s/88b8l05ufezn2yr/social_dis_ner_train_config_meta_social_dis_ner_production_samples.json?dl=0
curl -L -o ./preprocessed_data/social_dis_ner_test_config_meta_social_dis_ner_production_samples.json https://www.dropbox.com/s/65izcifcntxp0xr/social_dis_ner_test_config_meta_social_dis_ner_production_samples.json?dl=0
curl -L -o ./preprocessed_data/social_dis_ner_train_config_meta_social_dis_ner_dry_run_types.txt https://www.dropbox.com/s/ql3qh4swy92xv4p/social_dis_ner_train_config_meta_social_dis_ner_dry_run_types.txt?dl=0
