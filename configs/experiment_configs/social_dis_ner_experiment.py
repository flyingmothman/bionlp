from utils.config import AdafactorModifier, AdamModifier, EpochsCustomModifier, SmallerSpanWidthModifier, get_experiment_config, Epochs20Modifier, TestEveryEpochModifier


experiments = [
    # SPAN
    get_experiment_config(
        model_config_module_name='span_default',
        dataset_config_name='social_dis_ner_vanilla',
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
        dataset_config_name='social_dis_ner_vanilla',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            AdafactorModifier()
        ]
    ),

    # CRF
    get_experiment_config(
        model_config_module_name='seq_crf_default',
        dataset_config_name='social_dis_ner_vanilla',
        modifiers=[
            EpochsCustomModifier(num_epochs=25),
            TestEveryEpochModifier(),
            AdafactorModifier()
        ]
    ),
]
