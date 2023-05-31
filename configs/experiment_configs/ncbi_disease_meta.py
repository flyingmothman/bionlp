from utils.config import BiggerBatchModifier, EpochsCustomModifier, get_experiment_config, AccuracyEvaluationModifier, TestEveryEpochModifier, AdamModifier


experiments = [
    get_experiment_config(
        model_config_module_name='meta',
        dataset_config_name='ncbi_disease_meta',
        modifiers=[
            EpochsCustomModifier(num_epochs=15),
            TestEveryEpochModifier(),
            AdamModifier(),
            AccuracyEvaluationModifier(),
            BiggerBatchModifier(),
        ]
    ),
]
