from oe_eval.tasks.oe_eval_tasks.naturalqs_open import NaturalQsOpen


class NaturalQS_Base(NaturalQsOpen):
    MAX_TRAINING_DOCS = 30000

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            # Data is too large to fit in memory. We just sample from the first bit.
            # create a random shuffle and select first 1000 examples from sample TODO
            # select training docs excluding 1000 examples used for validation
            train_dataset = (
                self.dataset["train"]
                .shuffle(seed=0)
                .select(range(1000, len(self.dataset["train"])))
            )
            self._training_docs = list(map(self._process_doc, train_dataset))

        return self._training_docs

    def validation_docs(self):
        # select 1000 examples from the train set as validation set
        val_dataset = self.dataset["train"].shuffle(seed=0).select(range(0, 1000))
        return map(self._process_doc, val_dataset)

    def test_docs(self):
        # validation set used to be the test set by default if a test set did not exist, so we still set it as test set
        return map(self._process_doc, self.dataset["validation"])


class NaturalQS_Train_0shot(NaturalQS_Base):
    # we add an extra space
    def _process_doc(self, doc):
        # Q: In what year did the Motion Picture Production Code cease?
        # A: 1968

        q_prefix = "Q: " if self.task_config["context_kwargs"]["short_prefix"] else "Question: "
        a_prefix = "A:" if self.task_config["context_kwargs"]["short_prefix"] else "Answer: "
        if self.task_config["context_kwargs"]["reduced_spacing"]:
            query = f"Title: {doc['title']}\nBackground: {doc['context']}\n"
            query += f"{q_prefix}{doc['question']}\n{a_prefix}"
        else:
            query = f"Title: {doc['title']}\n\nBackground: {doc['context']}\n\n"
            query += f"{q_prefix}{doc['question']}\n\n{a_prefix}"
        answers_text = doc["answers"]["text"]
        out_doc = {
            "id": doc["id"],
            "title": doc["title"],
            "query": query,
            "answers_text": answers_text,
            "choices": [answers_text[0]],  # for perplexity eval
        }
        return out_doc

    # this shouldn't matter, but we remove the leading space since we added it into the query
    def doc_to_target(self, doc):
        return doc["answers_text"][0]


class NaturalQS_Validation_0shot(NaturalQS_Base):
    # we add an extra space
    def _process_doc(self, doc):
        # Q: In what year did the Motion Picture Production Code cease?
        # A: 1968

        q_prefix = "Q: " if self.task_config["context_kwargs"]["short_prefix"] else "Question: "
        a_prefix = "A:" if self.task_config["context_kwargs"]["short_prefix"] else "Answer: "
        if self.task_config["context_kwargs"]["reduced_spacing"]:
            query = f"Title: {doc['title']}\nBackground: {doc['context']}\n"
            query += f"{q_prefix}{doc['question']}\n{a_prefix}"
        else:
            query = f"Title: {doc['title']}\n\nBackground: {doc['context']}\n\n"
            query += f"{q_prefix}{doc['question']}\n\n{a_prefix}"
        answers_text = doc["answers"]["text"]
        out_doc = {
            "id": doc["id"],
            "title": doc["title"],
            "query": query,
            "answers_text": answers_text,
            "choices": [answers_text[0]],  # for perplexity eval
        }
        return out_doc

    # this shouldn't matter, but we remove the leading space since we added it into the query
    def doc_to_target(self, doc):
        return doc["answers_text"][0]


class NaturalQS_Test_0shot(NaturalQS_Base):
    pass
