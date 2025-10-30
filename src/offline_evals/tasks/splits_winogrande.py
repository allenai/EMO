from oe_eval.tasks.oe_eval_tasks.winogrande import Winogrande, WinograndeMC

class Winogrande_RC_BASE(Winogrande):
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

class Winogrande_MC_BASE(WinograndeMC):
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

class Winogrande_RC_Train(Winogrande_RC_BASE):
    pass

class Winogrande_RC_Validation(Winogrande_RC_BASE):
    pass

class Winogrande_RC_Test(Winogrande_RC_BASE):
    pass

class Winogrande_MC_Train(Winogrande_MC_BASE):
    pass

class Winogrande_MC_Validation(Winogrande_MC_BASE):
    pass

class Winogrande_MC_Test(Winogrande_MC_BASE):
    pass
