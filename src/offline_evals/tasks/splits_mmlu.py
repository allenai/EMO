from datasets import DatasetDict
from oe_eval.tasks.oe_eval_tasks.mmlu import GenericMMLU
from oe_eval.utilities.datasets_wrapper import MOUNTED_WEKA_DATASET_WRAPPER

MMLU_CATEGORIES = {
    "math": [
        "abstract_algebra",
        "college_mathematics",
        "elementary_mathematics",
        "high_school_mathematics",
        "high_school_statistics",
    ],
    "health": [
        "anatomy",
        "clinical_knowledge",
        "college_medicine",
        "human_aging",
        "medical_genetics",
        "nutrition",
        "professional_medicine",
        "virology",
    ],
    "physics": [
        "astronomy",
        "college_physics",
        "conceptual_physics",
        "high_school_physics",
    ],
    "business": [
        "business_ethics",
        "management",
        "marketing",
    ],
    "biology": [
        "college_biology",
        "high_school_biology",
    ],
    "chemistry": [
        "college_chemistry",
        "high_school_chemistry",
    ],
    "computer_science": [
        "college_computer_science",
        "computer_security",
        "high_school_computer_science",
        "machine_learning",
    ],
    "economics": [
        "econometrics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
    ],
    "engineering": [
        "electrical_engineering",
    ],
    "philosophy_cat": [
        "formal_logic",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "world_religions",
    ],
    "other": [
        "global_facts",
        "miscellaneous",
        "professional_accounting",
    ],
    "history": [
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "prehistory",
    ],
    "geography": [
        "high_school_geography",
    ],
    "politics": [
        "high_school_government_and_politics",
        "public_relations",
        "security_studies",
        "us_foreign_policy",
    ],
    "psychology": [
        "high_school_psychology",
        "professional_psychology",
    ],
    "culture": [
        "human_sexuality",
        "sociology",
    ],
    "law": [
        "international_law",
        "jurisprudence",
        "professional_law",
    ],
}

class GenericMMLU_withsplits(GenericMMLU):
    TEST_FRACTION = 0.6

    def validation_docs(self):
        return self.dataset["validation"].map(self._process_doc, with_indices=True)

    def test_docs(self):
        tot_test_size = len(self.dataset["test"])
        test_split = self.dataset["test"].shuffle(seed=0).select(range(int(tot_test_size * self.TEST_FRACTION), tot_test_size))
        return test_split.map(self._process_doc, with_indices=True)

    def training_docs(self):
        tot_test_size = len(self.dataset["test"])
        train_split = self.dataset["test"].shuffle(seed=0).select(range(0, int(tot_test_size * self.TEST_FRACTION)))
        return train_split.map(self._process_doc, with_indices=True)

class MMLU_17categories_RC(GenericMMLU):
    # choose from one of 17 categories using the task_config in tasks.py
    TEST_FRACTION = 0.6

    # override the download function to download from all MMLU tasks and choose accordingly
    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        # check that our chosen category is a valid category
        if self.task_config["category_name"] not in MMLU_CATEGORIES:
            raise ValueError(f"Category {self.task_config['category_name']} not recognized. Please choose from {list(MMLU_CATEGORIES.keys())}")

        dataset_names = MMLU_CATEGORIES[self.task_config["category_name"]]

        datasets = []

        for dataset_name in dataset_names:
            dataset = MOUNTED_WEKA_DATASET_WRAPPER.load_dataset(
                path=self.task_config["dataset_path"],
                name=dataset_name,
                data_dir=data_dir or self.data_dir,
                cache_dir=cache_dir or self.cache_dir,
                download_mode=download_mode or self.download_mode,
                revision=self.task_config.get("revision"),
                trust_remote_code=True,
            )
            datasets.append(dataset)

        # make sure all datasets have the same splits
        for ds in datasets[1:]:
            if ds.keys() != datasets[0].keys():
                raise ValueError("All datasets must have the same splits to be merged.")

        # merge the datasets
        from datasets import concatenate_datasets

        combined_dataset = DatasetDict()
        for split in datasets[0].keys():
            split_datasets = [ds[split] for ds in datasets]
            combined_dataset[split] = concatenate_datasets(split_datasets)

        self.dataset = combined_dataset

    def validation_docs(self):
        return self.dataset["validation"].map(self._process_doc, with_indices=True)

    def test_docs(self):
        tot_test_size = len(self.dataset["test"])
        test_split = self.dataset["test"].shuffle(seed=0).select(range(int(tot_test_size * self.TEST_FRACTION), tot_test_size))
        return test_split.map(self._process_doc, with_indices=True)

    def training_docs(self):
        tot_test_size = len(self.dataset["test"])
        train_split = self.dataset["test"].shuffle(seed=0).select(range(0, int(tot_test_size * self.TEST_FRACTION)))
        return train_split.map(self._process_doc, with_indices=True)

def create_mmlu_categories_tasks_withsplits(category):
    class MMLU_Category(MMLU_17categories_RC):
        DATASET_NAME = category
    return MMLU_Category

def create_mmlu_tasks_withsplits(subject):
    class MMLU(GenericMMLU_withsplits):
        DATASET_NAME = subject
    return MMLU