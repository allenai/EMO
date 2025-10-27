from oe_eval.tasks.oe_eval_tasks.arc import ARCEasyMC


class ARCEasyMC_Train(ARCEasyMC):
    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "ai2_arc",
        "dataset_name": "ARC-Easy",
        "native_id_field": "id",
        "primary_metric": "acc_raw",
        "split": "train",
    }

class ARCEasyMC_Validation(ARCEasyMC):
    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "ai2_arc",
        "dataset_name": "ARC-Easy",
        "native_id_field": "id",
        "primary_metric": "acc_raw",
        "split": "validation",
    }

class ARCEasyMC_Test(ARCEasyMC):
    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "ai2_arc",
        "dataset_name": "ARC-Easy",
        "native_id_field": "id",
        "primary_metric": "acc_raw",
        "split": "test",
    }