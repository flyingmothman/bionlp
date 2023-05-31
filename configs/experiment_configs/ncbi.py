from utils.config import AdafactorModifier, EpochsCustomModifier, SmallerSpanWidthModifier, get_experiment_config, Epochs20Modifier, TestEveryEpochModifier


experiments = [
    # SPAN
    get_experiment_config(
        model_config_module_name='span_default',
        dataset_config_name='ncbi_disease_sentence',
        modifiers=[
            Epochs20Modifier(),
            SmallerSpanWidthModifier(),
            TestEveryEpochModifier(),
            AdafactorModifier()
        ]
    ),

    # SEQ
    get_experiment_config(
        model_config_module_name='seq_default',
        dataset_config_name='ncbi_disease_sentence',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            AdafactorModifier()
        ]
    ),

    # CRF
    get_experiment_config(
        model_config_module_name='seq_crf_default',
        dataset_config_name='ncbi_disease_sentence',
        modifiers=[
            EpochsCustomModifier(num_epochs=25),
            TestEveryEpochModifier(),
            AdafactorModifier()
        ]
    ),
]
