from oe_eval.tasks.oe_eval_tasks.siqa import SocialIQA, SocialIQAMC

class SocialIQA_RC_Base(SocialIQA):
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

class SocialIQAMC_Base(SocialIQAMC):
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

class SocialIQA_RC_Train(SocialIQA_RC_Base):
    pass

class SocialIQA_RC_Validation(SocialIQA_RC_Base):
    pass

class SocialIQA_RC_Test(SocialIQA_RC_Base):
    pass

class SocialIQA_MC_Train(SocialIQAMC_Base):
    pass

class SocialIQA_MC_Validation(SocialIQAMC_Base):
    pass

class SocialIQA_MC_Test(SocialIQAMC_Base):
    pass
