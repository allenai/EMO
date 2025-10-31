from oe_eval.default_configs import MODEL_DEFAULTS as BASE_MODEL_DEFAULTS

# Extend the default configs with your additional parameters
MODEL_DEFAULTS = BASE_MODEL_DEFAULTS.copy()
MODEL_DEFAULTS.update(
    {
        "top_k": None,
        "weights_dir": None,
        "weight_calculation": None,
        "expert_model_0": None,
        "expert_model_1": None,
        "expert_model_2": None,
        "expert_model_3": None,
        "expert_model_4": None,
        "expert_model_5": None,
    }
)
