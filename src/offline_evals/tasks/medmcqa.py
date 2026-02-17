from oe_eval.tasks.oe_eval_tasks.medmcqa import MedMCQA as OriginalMedMCQA
from oe_eval.tasks.oe_eval_tasks.medmcqa import MedMCQAMC as OriginalMedMCQAMC
from oe_eval.tasks.utils import make_cloze_prompt, make_mcq_prompt


class MedMCQARC(OriginalMedMCQA):
    """RC-style: model scores full answer text continuations."""

    TASK_CONFIG_DEFAULTS = {
        **OriginalMedMCQA.TASK_CONFIG_DEFAULTS,
        "split": "validation",  # test split has hidden labels (cop=-1)
        "primary_metric": "acc_per_char",
    }

    def _process_doc(self, doc, index=-1):
        choices = [doc["opa"], doc["opb"], doc["opc"], doc["opd"]]
        query = make_cloze_prompt(doc["question"])
        out_doc = {
            "index": index,
            "query": query,
            "choices": choices,
            "gold": int(doc["cop"]) - 1,  # cop is 1-indexed (1=A,2=B,3=C,4=D)
        }
        return out_doc


class MedMCQAMC(OriginalMedMCQAMC):
    """MC-style: model scores letter label continuations (A/B/C/D)."""

    TASK_CONFIG_DEFAULTS = {
        **OriginalMedMCQAMC.TASK_CONFIG_DEFAULTS,
        "split": "validation",  # test split has hidden labels (cop=-1)
    }

    def _process_doc(self, doc, index=-1):
        choices = [doc["opa"], doc["opb"], doc["opc"], doc["opd"]]
        num_choices = len(choices)
        choice_labels = ["A", "B", "C", "D"][:num_choices]
        query = make_mcq_prompt(doc["question"], choices)
        out_doc = {
            "index": index,
            "query": query,
            "choices": choice_labels,
            "gold": int(doc["cop"]) - 1,  # cop is 1-indexed (1=A,2=B,3=C,4=D)
        }
        return out_doc
