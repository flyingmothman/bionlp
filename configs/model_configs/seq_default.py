from utils.config import ModelConfig, get_large_model_config

def create_model_config(model_config_module_name):
    model_config: ModelConfig = get_large_model_config(
        model_config_name=model_config_module_name,
        model_name='SeqDefault'
    )
    return model_config
