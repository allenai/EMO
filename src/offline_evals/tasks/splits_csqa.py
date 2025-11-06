from oe_eval.tasks.oe_eval_tasks.csqa import CommonsenseQA, CommonsenseQAMC

class CommonsenseQA_RC_Base(CommonsenseQA):
    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            # select training docs excluding 1000 examples used for validation
            train_dataset = self.dataset["train"].shuffle(seed=0).select(range(1000, len(self.dataset["train"])))
            self._training_docs = list(map(self._process_doc, train_dataset))
        return self._training_docs

    def validation_docs(self):
        # select 1000 examples from the train set as validation set
        val_dataset = self.dataset["train"].shuffle(seed=0).select(range(0, 1000))
        return map(self._process_doc, val_dataset)

    def test_docs(self):
        # validation set used to be the test set by default if a test set did not exist, so we still set it as test set
        return map(self._process_doc, self.dataset["validation"])

class CommonsenseQAMC_Base(CommonsenseQAMC):
    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            # select training docs excluding 1000 examples used for validation
            train_dataset = self.dataset["train"].shuffle(seed=0).select(range(1000, len(self.dataset["train"])))
            self._training_docs = list(map(self._process_doc, train_dataset))
        return self._training_docs

    def validation_docs(self):
        # select 1000 examples from the train set as validation set
        val_dataset = self.dataset["train"].shuffle(seed=0).select(range(0, 1000))
        return map(self._process_doc, val_dataset)

    def test_docs(self):
        # validation set used to be the test set by default if a test set did not exist, so we still set it as test set
        return map(self._process_doc, self.dataset["validation"])

class CommonsenseQA_RC_Train(CommonsenseQA_RC_Base):
    pass

class CommonsenseQA_RC_Validation(CommonsenseQA_RC_Base):
    pass

class CommonsenseQA_RC_Train_0shot(CommonsenseQA_RC_Base):
    pass

class CommonsenseQA_RC_Validation_0shot(CommonsenseQA_RC_Base):
    pass

class CommonsenseQA_RC_Test(CommonsenseQA_RC_Base):
    pass

class CommonsenseQA_MC_Train(CommonsenseQAMC_Base):
    pass

class CommonsenseQA_MC_Validation(CommonsenseQAMC_Base):
    pass

class CommonsenseQA_MC_Test(CommonsenseQAMC_Base):
    pass

