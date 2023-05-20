from utils.config import AdafactorModifier, AdamModifier, EpochsCustomModifier, SmallerSpanWidthModifier, get_experiment_config, Epochs20Modifier, TestEveryEpochModifier


experiments = [
    # SPAN
    get_experiment_config(
        model_config_module_name='model_span_large_default',
        dataset_config_name='social_dis_ner_vanilla',
        modifiers=[
            Epochs20Modifier(),
            SmallerSpanWidthModifier(),
            TestEveryEpochModifier(),
            AdafactorModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_default',
        dataset_config_name='social_dis_ner_vanilla',
        modifiers=[
            Epochs20Modifier(),
            SmallerSpanWidthModifier(),
            TestEveryEpochModifier(),
            AdamModifier()
        ]
    ),


    # SEQ
    get_experiment_config(
        model_config_module_name='model_seq_large_default',
        dataset_config_name='social_dis_ner_vanilla',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            AdafactorModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_default',
        dataset_config_name='social_dis_ner_vanilla',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            AdamModifier()
        ]
    ),


    # CRF
    get_experiment_config(
        model_config_module_name='model_seq_large_crf',
        dataset_config_name='social_dis_ner_vanilla',
        modifiers=[
            EpochsCustomModifier(num_epochs=25),
            TestEveryEpochModifier(),
            AdafactorModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_crf',
        dataset_config_name='social_dis_ner_vanilla',
        modifiers=[
            EpochsCustomModifier(num_epochs=25),
            TestEveryEpochModifier(),
            AdamModifier()
        ]
    ),
]
