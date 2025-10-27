from oe_eval.tasks.oe_eval_tasks.arc import ARCEasy


class ARCEasyMC_Train(ARCEasy):
    TASK_CONFIG_DEFAULTS: dict = {
        "split": "train",
        "dataset_path": "ai2_arc",
        "dataset_name": "ARC-Easy",
        "native_id_field": "id",
        "primary_metric": "acc_raw",
    }


class ARCEasyMC_Validation(ARCEasy):
    TASK_CONFIG_DEFAULTS: dict = {
        "split": "validation",
        "dataset_path": "ai2_arc",
        "dataset_name": "ARC-Easy",
        "native_id_field": "id",
        "primary_metric": "acc_raw",
    }

class ARCEasyMC_Test(ARCEasy):
    TASK_CONFIG_DEFAULTS: dict = {
        "split": "test",
        "dataset_path": "ai2_arc",
        "dataset_name": "ARC-Easy",
        "native_id_field": "id",
        "primary_metric": "acc_raw",
    }