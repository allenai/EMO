from oe_eval.tasks.oe_eval_tasks.coqa import CoQA


class COQA_Base(CoQA):
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

class COQA_Full_Base(CoQA):
    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            # select training docs excluding 1000 examples used for validation
            train_dataset = self.dataset["train"].shuffle(seed=0).select(range(1000, len(self.dataset["train"])))
            # self._training_docs = list(map(self._process_doc, train_dataset))
            self._training_docs = self._process_all_docs(train_dataset)
        return self._training_docs

    def validation_docs(self):
        # select 1000 examples from the train set as validation set
        val_dataset = self.dataset["train"].shuffle(seed=0).select(range(0, 1000))
        return self._process_all_docs(val_dataset)
        # return map(self._process_doc, val_dataset)

    def test_docs(self):
        # validation set used to be the test set by default if a test set did not exist, so we still set it as test set
        return self._process_all_docs(self.dataset["validation"])
        # return map(self._process_doc, self.dataset["validation"])

class COQA_full_Train_0shot(COQA_Base):
    pass

class COQA_full_Validation_0shot(COQA_Base):
    pass

class COQA_full_Test_0shot(COQA_Base):
    pass

class COQA_Train_0shot(COQA_Base):
    # # we override this to add a "space" after the "Answer" to match the same format.
    # def _process_doc_to_multi(self, doc):
    #     new_docs = []
    #     core_id = doc["id"]
    #     source = doc["source"]
    #     story = doc["story"]
    #     questions = doc["questions"]["input_text"]
    #     all_answers = doc["answers"]["input_text"]
    #     additional_answers = [x["input_text"] for x in doc["additional_answers"].values()]
    #     previous_qa: list = []
    #     for turn_idx, question in enumerate(questions):
    #         question = questions[turn_idx]
    #         answers = [all_answers[turn_idx]]
    #         for answer_list in additional_answers:
    #             if len(answer_list) > turn_idx and answer_list[turn_idx]:
    #                 answers.append(answer_list[turn_idx])
    #         query = f"Passage: {story}"
    #         if previous_qa:
    #             query += "\n\nPreceding questions:"
    #             for prev in previous_qa:
    #                 query += f"\n\nQuestion: {prev['question']}\nAnswer: {prev['answer']}"
    #         query += "\n\nFinal question:"
    #         query += f"\n\nQuestion: {question}\nAnswer: "
    #         new_doc = {
    #             "id": f"{core_id}_turn{turn_idx}",
    #             "source": source,
    #             "story": story,
    #             "query": query,
    #             "question": question,
    #             "answers": answers,
    #             "choices": [answers[0]],
    #             "previous_qa": previous_qa,
    #         }
    #         previous_qa.append({"question": question, "answer": answers[0]})
    #         new_docs.append(new_doc)
    #     return new_docs
    #
    # # this shouldn't matter, but we also change it anyways by removing the leading " "
    # def doc_to_target(self, doc):
    #     # return " " + doc["answers"][0]
    #     return doc["answers"][0]
    pass

class COQA_Validation_0shot(COQA_Base):
    # # we override this to add a "space" after the "Answer" to match the same format.
    # def _process_doc_to_multi(self, doc):
    #     new_docs = []
    #     core_id = doc["id"]
    #     source = doc["source"]
    #     story = doc["story"]
    #     questions = doc["questions"]["input_text"]
    #     all_answers = doc["answers"]["input_text"]
    #     additional_answers = [x["input_text"] for x in doc["additional_answers"].values()]
    #     previous_qa: list = []
    #     for turn_idx, question in enumerate(questions):
    #         question = questions[turn_idx]
    #         answers = [all_answers[turn_idx]]
    #         for answer_list in additional_answers:
    #             if len(answer_list) > turn_idx and answer_list[turn_idx]:
    #                 answers.append(answer_list[turn_idx])
    #         query = f"Passage: {story}"
    #         if previous_qa:
    #             query += "\n\nPreceding questions:"
    #             for prev in previous_qa:
    #                 query += f"\n\nQuestion: {prev['question']}\nAnswer: {prev['answer']}"
    #         query += "\n\nFinal question:"
    #         query += f"\n\nQuestion: {question}\nAnswer: "
    #         new_doc = {
    #             "id": f"{core_id}_turn{turn_idx}",
    #             "source": source,
    #             "story": story,
    #             "query": query,
    #             "question": question,
    #             "answers": answers,
    #             "choices": [answers[0]],
    #             "previous_qa": previous_qa,
    #         }
    #         previous_qa.append({"question": question, "answer": answers[0]})
    #         new_docs.append(new_doc)
    #     return new_docs
    #
    # # this shouldn't matter, but we also change it anyways by removing the leading " "
    # def doc_to_target(self, doc):
    #     # return " " + doc["answers"][0]
    #     return doc["answers"][0]
    pass

class COQA_Test_0shot(COQA_Base):
    # # we override this to add a "space" after the "Answer" to match the same format.
    # def _process_doc_to_multi(self, doc):
    #     new_docs = []
    #     core_id = doc["id"]
    #     source = doc["source"]
    #     story = doc["story"]
    #     questions = doc["questions"]["input_text"]
    #     all_answers = doc["answers"]["input_text"]
    #     additional_answers = [x["input_text"] for x in doc["additional_answers"].values()]
    #     previous_qa: list = []
    #     for turn_idx, question in enumerate(questions):
    #         question = questions[turn_idx]
    #         answers = [all_answers[turn_idx]]
    #         for answer_list in additional_answers:
    #             if len(answer_list) > turn_idx and answer_list[turn_idx]:
    #                 answers.append(answer_list[turn_idx])
    #         query = f"Passage: {story}"
    #         if previous_qa:
    #             query += "\n\nPreceding questions:"
    #             for prev in previous_qa:
    #                 query += f"\n\nQuestion: {prev['question']}\nAnswer: {prev['answer']}"
    #         query += "\n\nFinal question:"
    #         query += f"\n\nQuestion: {question}\nAnswer: "
    #         new_doc = {
    #             "id": f"{core_id}_turn{turn_idx}",
    #             "source": source,
    #             "story": story,
    #             "query": query,
    #             "question": question,
    #             "answers": answers,
    #             "choices": [answers[0]],
    #             "previous_qa": previous_qa,
    #         }
    #         previous_qa.append({"question": question, "answer": answers[0]})
    #         new_docs.append(new_doc)
    #     return new_docs
    #
    # # this shouldn't matter, but we also change it anyways by removing the leading " "
    # def doc_to_target(self, doc):
    #     # return " " + doc["answers"][0]
    #     return doc["answers"][0]
    pass