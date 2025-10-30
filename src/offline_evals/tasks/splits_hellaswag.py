from oe_eval.tasks.oe_eval_tasks.hellaswag import HellaSwag, HellaSwagMC

class HellaSwag_RC_Base(HellaSwag):
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

class HellaSwag_MC_Base(HellaSwagMC):
    def has_test_docs(self):
        return True

    def validation_docs(self):
        # Shuffle and select first half as validation set
        val_dataset = self.dataset["validation"].shuffle(seed=0).select(range(0, len(self.dataset["validation"]) // 2))
        return map(self._process_doc, val_dataset)

    def test_docs(self):
        # Shuffle and select second half as test set
        test_dataset = self.dataset["validation"].shuffle(seed=0).select(range(len(self.dataset["validation"]) // 2, len(self.dataset["validation"])))
        return map(self._process_doc, test_dataset)

class HellaSwag_RC_Train(HellaSwag_RC_Base):
    pass

class HellaSwag_RC_Validation(HellaSwag_RC_Base):
    pass

class HellaSwag_RC_Test(HellaSwag_RC_Base):
    pass

class HellaSwag_MC_Train(HellaSwag_MC_Base):
    pass

class HellaSwag_MC_Validation(HellaSwag_MC_Base):
    pass

class HellaSwag_MC_Test(HellaSwag_MC_Base):
    pass

