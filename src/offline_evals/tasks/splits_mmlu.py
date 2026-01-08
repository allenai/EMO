from oe_eval.tasks.oe_eval_tasks.mmlu import GenericMMLU

class GenericMMLU_withsplits(GenericMMLU):
    TEST_FRACTION = 0.6

    def validation_docs(self):
        return self.dataset["validation"].map(self._process_doc, with_indices=True)

    def test_docs(self):
        tot_test_size = len(self.dataset["test"])
        test_split = self.dataset["test"].shuffle(seed=0).select(range(int(tot_test_size * self.TEST_FRACTION), tot_test_size))
        return test_split.map(self._process_doc, with_indices=True)

    def training_docs(self):
        tot_test_size = len(self.dataset["test"])
        train_split = self.dataset["test"].shuffle(seed=0).select(range(0, int(tot_test_size * self.TEST_FRACTION)))
        return train_split.map(self._process_doc, with_indices=True)


def create_mmlu_tasks_withsplits(subject):
    class MMLU(GenericMMLU_withsplits):
        DATASET_NAME = subject
    return MMLU