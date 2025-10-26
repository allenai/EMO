from oe_eval.tasks.oe_eval_tasks.medmcqa import MedMCQAMC as OriginalMedMCQAMC
from oe_eval.tasks.utils import make_mcq_prompt


class MedMCQAMC(OriginalMedMCQAMC):
    def _process_doc(self, doc, index=-1):
        choices = [doc["opa"], doc["opb"], doc["opc"], doc["opd"]]
        num_choices = len(choices)
        choice_labels = ["A", "B", "C", "D"][:num_choices]
        query = make_mcq_prompt(doc["question"], choices)
        out_doc = {
            "index": index,
            "query": query,
            "choices": choice_labels,
            "gold": int(doc["cop"]),
        }
        return out_doc
