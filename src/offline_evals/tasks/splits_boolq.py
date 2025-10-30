from oe_eval.tasks.oe_eval_tasks.boolq import BoolQ, BoolQMC

class BoolQ_RC_Base(BoolQ):
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


class BoolQ_MC_Base(BoolQMC):
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

class BoolQ_RC_Train(BoolQ_RC_Base):
    pass

class BoolQ_RC_Validation(BoolQ_RC_Base):
    pass

class BoolQ_RC_Test(BoolQ_RC_Base):
    pass

class BoolQ_MC_Train(BoolQ_MC_Base):
    pass

class BoolQ_MC_Validation(BoolQ_MC_Base):
    pass

class BoolQ_MC_Test(BoolQ_MC_Base):
    pass

