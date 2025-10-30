from oe_eval.tasks.oe_eval_tasks.piqa import PiQA, PiQAMC

class PIQA_RC_BASE(PiQA):
    def has_test_docs(self):
        return True

    def validation_docs(self):
        # Shuffle and select second half as validation set
        val_dataset = self.dataset["validation"].shuffle(seed=0).select(range(len(self.dataset["validation"]) // 2, len(self.dataset["validation"])))
        return map(self._process_doc, val_dataset)

    def test_docs(self):
        # Shuffle and select first half as test set
        test_dataset = (self.dataset["validation"].shuffle(seed=0).select(range(0, len(self.dataset["validation"]) // 2)))
        return map(self._process_doc, test_dataset)

class PIQA_MC_BASE(PiQAMC):
    def has_test_docs(self):
        return True

    def validation_docs(self):
        # Shuffle and select first half as validation set
        val_dataset = self.dataset["validation"].shuffle(seed=0).select(range(0, len(self.dataset["validation"]) // 2))
        return map(self._process_doc, val_dataset)

    def test_docs(self):
        # Shuffle and select second half as test set
        test_dataset = self.dataset["validation"].shuffle(seed=0).select(
            range(len(self.dataset["validation"]) // 2, len(self.dataset["validation"])))
        return map(self._process_doc, test_dataset)

class PIQA_RC_Train(PIQA_RC_BASE):
    pass

class PIQA_RC_Validation(PIQA_RC_BASE):
    pass

class PIQA_RC_Test(PIQA_RC_BASE):
    pass

class PIQA_MC_Train(PIQA_MC_BASE):
    pass

class PIQA_MC_Validation(PIQA_MC_BASE):
    pass

class PIQA_MC_Test(PIQA_MC_BASE):
    pass

