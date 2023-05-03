import util
import train_util
import transformers
from transformers import AutoTokenizer
import time
import numpy as np
import logging  # configured in args.py
import importlib
from utils.config import DatasetConfig, ModelConfig, ExperimentConfig
from random import shuffle
from structs import DatasetSplit
from preamble import *

# Run some checks on our config files before starting
util.check_config_integrity()

# Get config for training
training_args = train_util.parse_training_args()
EXPERIMENT_NAME = training_args.experiment_name
IS_DRY_RUN = training_args.is_dry_run_mode
IS_TESTING = training_args.is_testing

experiments_module = importlib.import_module(f"experiments.{EXPERIMENT_NAME}")
experiments: list[ExperimentConfig] = experiments_module.experiments

# Setup logging
root_logger = logging.getLogger()
roots_handler = root_logger.handlers[0]
roots_handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))  # change formatting
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
logging.getLogger('dropbox').setLevel(logging.WARN)
transformers.logging.set_verbosity_error()

# Create directories where we will store training results
logger.info("Create folders for training")

training_results_folder_path = './training_results'

mistakes_folder_path = f'{training_results_folder_path}/mistakes'
error_visualization_folder_path = f'{training_results_folder_path}/error_visualizations'
predictions_folder_path = f'{training_results_folder_path}/predictions'
models_folder_path = f'{training_results_folder_path}/models'
performance_folder_path = f'{training_results_folder_path}/performance'
test_predictions_folder_path = f'{training_results_folder_path}/test_predictions'

util.create_directory_structure(mistakes_folder_path)
util.create_directory_structure(error_visualization_folder_path)
util.create_directory_structure(predictions_folder_path)
util.create_directory_structure(models_folder_path)
util.create_directory_structure(performance_folder_path)
util.create_directory_structure(test_predictions_folder_path)

validation_performance_file_path = f"{performance_folder_path}/performance_{EXPERIMENT_NAME}" \
                                   f"_{DatasetSplit.valid.name}.csv"
test_performance_file_path = f"{performance_folder_path}/performance_{EXPERIMENT_NAME}_{DatasetSplit.test.name}.csv"
train_util.create_performance_file_header(validation_performance_file_path)
train_util.create_performance_file_header(test_performance_file_path)

dataset_config: DatasetConfig
model_config: ModelConfig

# How frequently we will evaluate on test data

for experiment_idx, experiment_config in enumerate(experiments):
    dataset_config = experiment_config.dataset_config
    model_config = experiment_config.model_config
    test_evaluation_frequency = experiment_config.testing_frequency

    train_util.print_experiment_info(
        experiment_config=experiment_config,
        dataset_config=dataset_config,
        model_config=model_config,
        experiment_name=EXPERIMENT_NAME,
        is_dry_run=IS_DRY_RUN,
        is_testing=IS_TESTING,
        test_evaluation_frequency=test_evaluation_frequency
    )

    dataset_name = dataset_config.dataset_name

    # -------- READ DATA ---------
    logger.info("Starting to read data.")

    
    train_samples = train_util.get_train_samples(dataset_config)
    assert len(train_samples) == experiment_config.dataset_config.expected_number_of_train_samples,\
            f"expected num train samples: {experiment_config.dataset_config.expected_number_of_train_samples}"\
            f"but got {len(train_samples)}"

    valid_samples = train_util.get_valid_samples(dataset_config)
    assert len(valid_samples) == experiment_config.dataset_config.expected_number_of_valid_samples,\
            f"expected num valid samples: {experiment_config.dataset_config.expected_number_of_valid_samples}"\
            f"but got {len(valid_samples)}"

    if IS_TESTING:
        test_samples = train_util.get_test_samples(dataset_config)
        assert len(test_samples) == experiment_config.dataset_config.expected_number_of_test_samples,\
                f"expected num test samples: {experiment_config.dataset_config.expected_number_of_test_samples}"\
                f"but got {len(test_samples)}"
 
    # Do some important checks on the data
    if train_util.has_external_features(train_samples):
        assert model_config.external_feature_type is not None, \
                f"model is not expecting external features \n{model_config}"
        train_util.check_external_features(
                train_samples,
                model_config.external_feature_type)

    

    logger.info(f"num train samples: {len(train_samples)}")
    logger.info(f"num valid samples: {len(valid_samples)}")
    if IS_TESTING:
        logger.info(f"num test samples: {len(test_samples)}")

    if IS_DRY_RUN:  # during dry run, only work with the first 10 samples
        train_samples = train_samples[:20]
        valid_samples = valid_samples[:20]
        if IS_TESTING:
            test_samples = test_samples[:20]

    logger.info("finished reading data.")

    # Check samples
    util.ensure_no_sample_gets_truncated_by_bert(train_samples, dataset_config)

    # ------ MODEL INITIALISATION --------
    logger.info("Starting model initialization.")
    bert_tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name)
    model = train_util.prepare_model(model_config, dataset_config)
    optimizer = train_util.get_optimizer(model, experiment_config)
    all_types = util.get_all_types(dataset_config.types_file_path, dataset_config.num_types)
    logger.debug(f"all types\n {util.p_string(list(all_types))}")
    logger.info("Finished model initialization.")

    # verify that all label types in annotations are valid types
    train_util.check_label_types(train_samples, valid_samples, all_types)

    for epoch in range(experiment_config.num_epochs):
        # Don't dry run for more than 2 epochs while testing
        if IS_DRY_RUN and epoch >= 2:
            break

        epoch_loss = []

        # Begin Training
        logger.info("-" * 20)
        logger.info(f"Train epoch {epoch}")
        train_start_time = time.time()

        model.train()

        # shuffle samples every epoch
        shuffle(train_samples)

        # Training Loop
        for train_batch in train_util.get_batches(samples=train_samples, batch_size=model_config.batch_size):
            optimizer.zero_grad()
            loss, predicted_annos = model(train_batch)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.cpu().detach().numpy())
        logger.info(
            f"Epoch {epoch} " +
            magenta("Loss: " + f"{np.array(epoch_loss).mean()},") +
            f"Training Time: {str(time.time() - train_start_time)} " +
            f"seconds")
        logger.info(f"Done training epoch {epoch}")

        train_util.evaluate_validation_split(
            logger=logger,
            model=model,
            validation_samples=valid_samples,
            mistakes_folder_path=mistakes_folder_path,
            predictions_folder_path=predictions_folder_path,
            error_visualization_folder_path=error_visualization_folder_path,
            validation_performance_file_path=validation_performance_file_path,
            experiment_name=EXPERIMENT_NAME,
            model_config_name=model_config.model_config_name,
            dataset_config_name=dataset_config.dataset_config_name,
            epoch=epoch,
            experiment_idx=experiment_idx,
            evaluation_type=experiment_config.evaluation_type
        )

        if IS_TESTING:
            if (((epoch + 1) % test_evaluation_frequency) == 0) \
                    or IS_DRY_RUN:
                train_util.evaluate_test_split(
                    logger=logger,
                    model=model,
                    test_samples=test_samples,
                    mistakes_folder_path=mistakes_folder_path,
                    predictions_folder_path=predictions_folder_path,
                    error_visualization_folder_path=error_visualization_folder_path,
                    test_performance_file_path=test_performance_file_path,
                    experiment_name=EXPERIMENT_NAME,
                    model_config_name=model_config.model_config_name,
                    dataset_config_name=dataset_config.dataset_config_name,
                    epoch=epoch,
                    experiment_idx=experiment_idx,
                    evaluation_type=experiment_config.evaluation_type
                )

logger.info(green("Experiment Finished!!"))
