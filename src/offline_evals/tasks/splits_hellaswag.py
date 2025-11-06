from oe_eval.tasks.oe_eval_tasks.hellaswag import HellaSwag, HellaSwagMC

class HellaSwag_RC_Base(HellaSwag):
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

class HellaSwag_MC_Base(HellaSwagMC):
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

class HellaSwag_RC_Train(HellaSwag_RC_Base):
    pass

class HellaSwag_RC_Validation(HellaSwag_RC_Base):
    pass

class HellaSwag_RC_Train_0shot(HellaSwag_RC_Base):
    pass

class HellaSwag_RC_Validation_0shot(HellaSwag_RC_Base):
    pass

class HellaSwag_RC_Test(HellaSwag_RC_Base):
    pass

class HellaSwag_MC_Train(HellaSwag_MC_Base):
    pass

class HellaSwag_MC_Validation(HellaSwag_MC_Base):
    pass

class HellaSwag_MC_Test(HellaSwag_MC_Base):
    pass

