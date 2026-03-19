from datasets import DatasetDict
from oe_eval.tasks.oe_eval_tasks.mmlu import GenericMMLU
from oe_eval.utilities.datasets_wrapper import MOUNTED_WEKA_DATASET_WRAPPER

from ..metrics.mc_softloss import SoftLoss

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

    def make_metrics(self):
        # run the super
        super().make_metrics()
        # add softloss metric
        self._metrics += [SoftLoss(**self.task_config["metric_kwargs"])]

        return self._metrics

def _load_and_split_subjects(task_self, categories_dict, data_dir=None, cache_dir=None, download_mode=None):
    """Load MMLU subjects and split each subject's test set independently before concatenating.

    Each subject's HF "test" split is shuffled (seed=0) and split 60/40 into train/test.
    The results are concatenated across subjects, so the category's train set = union of
    per-subject train sets, and likewise for test. This ensures per-subject evals use the
    exact same test examples as the category eval (no train-test leakage).

    The HF "validation" and other splits (e.g. "dev", "auxiliary_train") are concatenated as-is.
    """
    from datasets import concatenate_datasets

    category_name = task_self.task_config["category_name"]
    if category_name not in categories_dict:
        raise ValueError(
            f"Category {category_name} not recognized. "
            f"Please choose from {list(categories_dict.keys())}"
        )

    dataset_names = categories_dict[category_name]
    test_fraction = task_self.TEST_FRACTION

    all_train_splits = []
    all_test_splits = []
    all_other_splits = {}  # split_name -> list of datasets

    for dataset_name in dataset_names:
        dataset = MOUNTED_WEKA_DATASET_WRAPPER.load_dataset(
            path=task_self.task_config["dataset_path"],
            name=dataset_name,
            data_dir=data_dir or task_self.data_dir,
            cache_dir=cache_dir or task_self.cache_dir,
            download_mode=download_mode or task_self.download_mode,
            revision=task_self.task_config.get("revision"),
            trust_remote_code=True,
        )

        # Split this subject's test set independently
        subj_test = dataset["test"]
        n = len(subj_test)
        subj_shuffled = subj_test.shuffle(seed=0)
        cutoff = int(n * test_fraction)
        all_train_splits.append(subj_shuffled.select(range(0, cutoff)))
        all_test_splits.append(subj_shuffled.select(range(cutoff, n)))

        # Collect non-test splits
        for split_name in dataset.keys():
            if split_name == "test":
                continue
            all_other_splits.setdefault(split_name, []).append(dataset[split_name])

    combined = DatasetDict()
    combined["train"] = concatenate_datasets(all_train_splits)
    combined["test"] = concatenate_datasets(all_test_splits)
    for split_name, split_list in all_other_splits.items():
        combined[split_name] = concatenate_datasets(split_list)

    task_self.dataset = combined


class MMLU_17categories_RC(GenericMMLU):
    # choose from one of 17 categories using the task_config in tasks.py
    TEST_FRACTION = 0.6

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        _load_and_split_subjects(self, MMLU_CATEGORIES, data_dir, cache_dir, download_mode)

    def validation_docs(self):
        return self.dataset["validation"].map(self._process_doc, with_indices=True)

    def test_docs(self):
        return self.dataset["test"].map(self._process_doc, with_indices=True)

    def training_docs(self):
        return self.dataset["train"].map(self._process_doc, with_indices=True)

    def make_metrics(self):
        # run the super
        super().make_metrics()
        # add softloss metric
        self._metrics += [SoftLoss(**self.task_config["metric_kwargs"])]

        return self._metrics

def create_mmlu_categories_tasks_withsplits(category):
    class MMLU_Category(MMLU_17categories_RC):
        DATASET_NAME = category
    return MMLU_Category

def create_mmlu_tasks_withsplits(subject):
    class MMLU(GenericMMLU_withsplits):
        DATASET_NAME = subject
    return MMLU


# ---------------------------------------------------------------------------
# Router-clustering-based categories (16 clusters from k-means on router
# embeddings, model: randpool, embedding: topk_freq, transform: mean_pca_l2)
# See claude_outputs/analysis/router_clustering_mmlu_val/mmlu_categories_match.md
# ---------------------------------------------------------------------------

MMLU_CLUSTER_CATEGORIES = {
    "cluster_chemistry": [
        "college_chemistry",
        "high_school_chemistry",
    ],
    "cluster_security_sociology": [
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
    "cluster_moral_scenarios": [
        "moral_scenarios",
    ],
    "cluster_psychology": [
        "professional_psychology",
        "high_school_psychology",
    ],
    "cluster_law_gov": [
        "professional_law",
        "high_school_government_and_politics",
        "international_law",
    ],
    "cluster_prehistory_religions": [
        "prehistory",
        "world_religions",
    ],
    "cluster_biomedical": [
        "anatomy",
        "clinical_knowledge",
        "college_medicine",
        "human_aging",
        "medical_genetics",
        "nutrition",
        "professional_medicine",
        "virology",
        "college_biology",
        "high_school_biology",
        "human_sexuality",
    ],
    "cluster_econ_geography": [
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_geography",
        "global_facts",
    ],
    "cluster_philosophy": [
        "moral_disputes",
        "philosophy",
        "logical_fallacies",
        "jurisprudence",
    ],
    "cluster_quantitative": [
        "elementary_mathematics",
        "high_school_mathematics",
        "high_school_statistics",
        "econometrics",
        "college_mathematics",
        "machine_learning",
    ],
    "cluster_accounting": [
        "professional_accounting",
    ],
    "cluster_miscellaneous": [
        "miscellaneous",
    ],
    "cluster_history": [
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
    ],
    "cluster_business": [
        "marketing",
        "public_relations",
        "business_ethics",
        "management",
    ],
    "cluster_physics_eng": [
        "conceptual_physics",
        "high_school_physics",
        "astronomy",
        "electrical_engineering",
        "college_physics",
    ],
    "cluster_cs_logic": [
        "formal_logic",
        "abstract_algebra",
        "college_computer_science",
        "computer_security",
        "high_school_computer_science",
    ],
}


class MMLU_16clusters_RC(GenericMMLU):
    """MMLU task that groups subjects by router-based clustering (16 clusters)."""
    TEST_FRACTION = 0.6

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        _load_and_split_subjects(self, MMLU_CLUSTER_CATEGORIES, data_dir, cache_dir, download_mode)

    def validation_docs(self):
        return self.dataset["validation"].map(self._process_doc, with_indices=True)

    def test_docs(self):
        return self.dataset["test"].map(self._process_doc, with_indices=True)

    def training_docs(self):
        return self.dataset["train"].map(self._process_doc, with_indices=True)

    def make_metrics(self):
        super().make_metrics()
        self._metrics += [SoftLoss(**self.task_config["metric_kwargs"])]
        return self._metrics


def create_mmlu_cluster_tasks_withsplits(cluster_name):
    class MMLU_Cluster(MMLU_16clusters_RC):
        DATASET_NAME = cluster_name
    return MMLU_Cluster