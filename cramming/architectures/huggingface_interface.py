"""HF model variations based on reconfiguring their huggingface implementations."""

import transformers


def construct_huggingface_model(cfg_arch, vocab_size):
    """construct model from given configuration. Only works if this arch exists on the hub."""

    if isinstance(cfg_arch, transformers.PretrainedConfig):
        configuration = cfg_arch
    else:
        model_type = cfg_arch["model_type"]
        configuration = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path=model_type, **cfg_arch)
    configuration.vocab_size = vocab_size
    model = transformers.AutoModelForPreTraining.from_config(configuration)
    model.vocab_size = model.config.vocab_size

    old_forward = model.forward

    def modified_forward(input_ids, attention_mask=None, **kwargs):
        return old_forward(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)

    model.forward = modified_forward

    return model
