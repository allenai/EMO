from oe_eval.tasks.oe_eval_tasks.csqa import CommonsenseQA, CommonsenseQAMC

class CommonsenseQA_RC_Base(CommonsenseQA):
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

class CommonsenseQAMC_Base(CommonsenseQAMC):
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

class CommonsenseQA_RC_Train(CommonsenseQA_RC_Base):
    pass

class CommonsenseQA_RC_Validation(CommonsenseQA_RC_Base):
    pass

class CommonsenseQA_RC_Test(CommonsenseQA_RC_Base):
    pass

class CommonsenseQA_MC_Train(CommonsenseQAMC_Base):
    pass

class CommonsenseQA_MC_Validation(CommonsenseQAMC_Base):
    pass

class CommonsenseQA_MC_Test(CommonsenseQAMC_Base):
    pass

