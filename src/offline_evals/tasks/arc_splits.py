from oe_eval.tasks.oe_eval_tasks.arc import ARCEasyMC, ARCEasy

class ARCEasyRC_Train(ARCEasy):
    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "ai2_arc",
        "dataset_name": "ARC-Easy-RC-Train",
        "native_id_field": "id",  # Field in doc that corresponds to the native id
        "primary_metric": "acc_per_char",
        "split": "train",  # Which split to evaluate on
        "context_kwargs": {
            "description": None,  # E.g., "The following are multiple choice questions with answers.\n\n",
        },
    }

class ARCEasyRC_Validation(ARCEasy):
    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "ai2_arc",
        "dataset_name": "ARC-Easy-RC-Validation",
        "native_id_field": "id",  # Field in doc that corresponds to the native id
        "primary_metric": "acc_per_char",
        "split": "validation",  # Which split to evaluate on
        "context_kwargs": {
            "description": None,  # E.g., "The following are multiple choice questions with answers.\n\n",
        },
    }

class ARCEasyRC_Test(ARCEasy):
    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "ai2_arc",
        "dataset_name": "ARC-Easy-RC-Test",
        "native_id_field": "id",  # Field in doc that corresponds to the native id
        "primary_metric": "acc_per_char",
        "split": "test",  # Which split to evaluate on
        "context_kwargs": {
            "description": None,  # E.g., "The following are multiple choice questions with answers.\n\n",
        },
    }


class ARCEasyMC_Train(ARCEasyMC):
    TASK_CONFIG_DEFAULTS: dict = {
        "split": "train",
        "dataset_path": "ai2_arc",
        "dataset_name": "ARC-Easy-MC-Train",
        "native_id_field": "id",
        "primary_metric": "acc_raw",
    }


class ARCEasyMC_Validation(ARCEasyMC):
    TASK_CONFIG_DEFAULTS: dict = {
        "split": "validation",
        "dataset_path": "ai2_arc",
        "dataset_name": "ARC-Easy-MC-Validation",
        "native_id_field": "id",
        "primary_metric": "acc_raw",
    }

class ARCEasyMC_Test(ARCEasyMC):
    TASK_CONFIG_DEFAULTS: dict = {
        "split": "test",
        "dataset_path": "ai2_arc",
        "dataset_name": "ARC-Easy-MC-Test",
        "native_id_field": "id",
        "primary_metric": "acc_raw",
    }