from datasets import Dataset, DatasetDict
from oe_eval.tasks.oe_eval_tasks.mmlu_pro import GenericMMLUProRC
from oe_eval.utilities.datasets_wrapper import MOUNTED_WEKA_DATASET_WRAPPER


# Map underscore task names -> original MMLU-Pro category names (for filtering)
MMLU_PRO_CATEGORIES_MAP = {
    "math": "math",
    "health": "health",
    "physics": "physics",
    "business": "business",
    "biology": "biology",
    "chemistry": "chemistry",
    "computer_science": "computer science",
    "economics": "economics",
    "engineering": "engineering",
    "philosophy": "philosophy",
    "other": "other",
    "history": "history",
    "psychology": "psychology",
    "law": "law",
}

# Number of examples per category reserved for pruning (router activation collection)
PRUNE_SIZE = 100

# Fraction of remaining examples (after pruning holdout) used for evaluation
TEST_FRACTION = 0.6


class GenericMMLUPro_withsplits(GenericMMLUProRC):
    """MMLU-Pro task with custom train/validation/test splits for the pruning pipeline.

    The upstream MMLU-Pro dataset has only validation (70 examples, 5/category) and
    test (12k examples). We repartition each category's test examples into:
      - validation: 100 examples for pruning (router activation collection)
      - train: 40% of remainder for finetuning
      - test: 60% of remainder for evaluation
    The original validation split (5/category) is kept as "dev" for few-shot demos.
    """

    CATEGORY_KEY: str = ""  # underscore name, e.g. "computer_science"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        full_dataset = MOUNTED_WEKA_DATASET_WRAPPER.load_dataset(
            path="TIGER-Lab/MMLU-Pro",
            name=None,
            data_dir=data_dir or self.data_dir,
            cache_dir=cache_dir or self.cache_dir,
            download_mode=download_mode or self.download_mode,
            revision=self.task_config.get("revision"),
            trust_remote_code=True,
        )

        original_category = MMLU_PRO_CATEGORIES_MAP[self.CATEGORY_KEY]

        # Filter to this category
        cat_test = full_dataset["test"].filter(
            lambda doc: doc["category"] == original_category
        )
        cat_dev = full_dataset["validation"].filter(
            lambda doc: doc["category"] == original_category
        )

        # Shuffle and split the test set
        n = len(cat_test)
        cat_shuffled = cat_test.shuffle(seed=0)

        prune_size = min(PRUNE_SIZE, n)
        prune_split = cat_shuffled.select(range(prune_size))

        remaining = cat_shuffled.select(range(prune_size, n))
        n_remaining = len(remaining)
        test_cutoff = int(n_remaining * TEST_FRACTION)

        test_split = remaining.select(range(test_cutoff))
        train_split = remaining.select(range(test_cutoff, n_remaining))

        self.dataset = DatasetDict({
            "dev": cat_dev,          # 5 examples for few-shot demos
            "validation": prune_split,  # 100 examples for pruning
            "train": train_split,       # ~40% of remainder for finetuning
            "test": test_split,         # ~60% of remainder for evaluation
        })

    def fewshot_examples(self, k, rnd, doc):
        # Use original validation (now "dev") for few-shot demos, matching upstream behavior
        if self._fewshot_docs is None:
            self._fewshot_docs = [
                self._process_doc(doc) for doc in self.dataset["dev"]
            ]
        return self._fewshot_docs[:k]

    def validation_docs(self):
        return self.dataset["validation"].map(self._process_doc, with_indices=True)

    def test_docs(self):
        return self.dataset["test"].map(self._process_doc, with_indices=True)

    def training_docs(self):
        return self.dataset["train"].map(self._process_doc, with_indices=True)


def create_mmlu_pro_category_tasks_withsplits(category_key):
    class MMLUPro_Category(GenericMMLUPro_withsplits):
        CATEGORY_KEY = category_key
        CATEGORY = MMLU_PRO_CATEGORIES_MAP[category_key]

    return MMLUPro_Category


class GenericMMLUPro_merged(GenericMMLUPro_withsplits):
    """MMLU-Pro variant where pruning and finetuning use the same merged data.

    Uses the same test set as GenericMMLUPro_withsplits (same shuffle seed and
    TEST_FRACTION), but merges the pruning and train portions into a single
    "train" split used for both pipeline phases:
      - validation/train: merged set (100 prune + 40% remainder) for both pruning and finetuning
      - test: 60% of remainder for evaluation (identical to the non-merged variant)
    """

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        from datasets import concatenate_datasets

        full_dataset = MOUNTED_WEKA_DATASET_WRAPPER.load_dataset(
            path="TIGER-Lab/MMLU-Pro",
            name=None,
            data_dir=data_dir or self.data_dir,
            cache_dir=cache_dir or self.cache_dir,
            download_mode=download_mode or self.download_mode,
            revision=self.task_config.get("revision"),
            trust_remote_code=True,
        )

        original_category = MMLU_PRO_CATEGORIES_MAP[self.CATEGORY_KEY]

        cat_test = full_dataset["test"].filter(
            lambda doc: doc["category"] == original_category
        )
        cat_dev = full_dataset["validation"].filter(
            lambda doc: doc["category"] == original_category
        )

        # Same shuffle + split logic as the non-merged variant to keep test set identical
        n = len(cat_test)
        cat_shuffled = cat_test.shuffle(seed=0)

        prune_size = min(PRUNE_SIZE, n)
        prune_split = cat_shuffled.select(range(prune_size))

        remaining = cat_shuffled.select(range(prune_size, n))
        n_remaining = len(remaining)
        test_cutoff = int(n_remaining * TEST_FRACTION)

        test_split = remaining.select(range(test_cutoff))
        train_split = remaining.select(range(test_cutoff, n_remaining))

        # Merge prune + train into a single set
        merged_train = concatenate_datasets([prune_split, train_split])

        self.dataset = DatasetDict({
            "dev": cat_dev,              # 5 examples for few-shot demos
            "validation": merged_train,  # merged set for pruning
            "train": merged_train,       # same merged set for finetuning
            "test": test_split,          # 60% of remainder (identical to non-merged)
        })


def create_mmlu_pro_merged_tasks_withsplits(category_key):
    class MMLUPro_Merged(GenericMMLUPro_merged):
        CATEGORY_KEY = category_key
        CATEGORY = MMLU_PRO_CATEGORIES_MAP[category_key]

    return MMLUPro_Merged
