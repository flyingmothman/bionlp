import transformers
import torch
from utils.structs import ExperimentConfig

def get_optimizer(model, experiment_config: ExperimentConfig):
    if experiment_config.optimizer == 'Ranger':
        # return torch_optimizer.Ranger(model.parameters(), dataset_config['learning_rate'])
        raise Exception("no ranger optimizer")
    elif experiment_config.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), experiment_config.model_config.learning_rate)
    elif experiment_config.optimizer == 'AdamW':
        return torch.optim.AdamW(model.parameters(), experiment_config.model_config.learning_rate)
    elif experiment_config.optimizer == 'Adafactor':
        return transformers.Adafactor(
            model.parameters(),
            lr=experiment_config.model_config.learning_rate,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    else:
        raise Exception(f"optimizer not found: {experiment_config.optimizer}")
