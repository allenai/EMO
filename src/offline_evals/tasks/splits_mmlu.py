from datasets import DatasetDict, concatenate_datasets
from oe_eval.tasks.base_task import Task
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
        test_split = (
            self.dataset["test"]
            .shuffle(seed=0)
            .select(range(int(tot_test_size * self.TEST_FRACTION), tot_test_size))
        )
        return test_split.map(self._process_doc, with_indices=True)

    def training_docs(self):
        tot_test_size = len(self.dataset["test"])
        train_split = (
            self.dataset["test"]
            .shuffle(seed=0)
            .select(range(0, int(tot_test_size * self.TEST_FRACTION)))
        )
        return train_split.map(self._process_doc, with_indices=True)

    def make_metrics(self):
        # run the super
        super().make_metrics()
        # add softloss metric
        self._metrics += [SoftLoss(**self.task_config["metric_kwargs"])]

        return self._metrics


def _load_and_split_subjects(
    task_self, categories_dict, data_dir=None, cache_dir=None, download_mode=None
):
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
    all_other_splits: dict[str, list] = {}  # split_name -> list of datasets

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


class _MMLU_PerSubjectContext_RC(GenericMMLU):
    """Base class for MMLU category tasks that use per-subject prompt context.

    For each question, the prompt header says "about {subject}" (not the category name)
    and the few-shot examples come from that subject's dev split. This ensures the
    category eval produces results identical to running per-subject evals independently,
    so the weighted per-subject average exactly matches the category micro-average.
    """

    TEST_FRACTION = 0.6

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        raise NotImplementedError("Subclasses must implement download()")

    def _process_doc(self, doc, index=-1):
        out_doc = super()._process_doc(doc, index=index)
        out_doc["subject"] = doc["subject"]
        return out_doc

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        subject = doc.get("subject", self.DATASET_NAME)
        description = f"The following are multiple choice questions (with answers) about {self._format_subject(subject)}.\n\n"

        if (
            "description" in self.task_config["context_kwargs"]
            and self.task_config["context_kwargs"]["description"] is not None
        ):
            description = self.task_config["context_kwargs"]["description"]

        kwargs["description"] = description
        # Call Task.fewshot_context directly, skipping GenericMMLU_MC.fewshot_context
        # which would overwrite description with self.DATASET_NAME
        return Task.fewshot_context(self, doc=doc, num_fewshot=num_fewshot, **kwargs)

    def fewshot_examples(self, k, rnd, doc):
        if self._fewshot_docs is None:
            self._fewshot_docs: dict[str, list] = {}
            for dev_doc in self.dataset["dev"]:
                subject = dev_doc["subject"]
                if subject not in self._fewshot_docs:
                    self._fewshot_docs[subject] = []
                self._fewshot_docs[subject].append(self._process_doc(dev_doc))

        subject = doc.get("subject", self.DATASET_NAME)
        subject_docs = self._fewshot_docs.get(subject, [])
        return subject_docs[:k]

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


class MMLU_17categories_RC(_MMLU_PerSubjectContext_RC):
    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        _load_and_split_subjects(self, MMLU_CATEGORIES, data_dir, cache_dir, download_mode)


def create_mmlu_categories_tasks_withsplits(category):
    class MMLU_Category(MMLU_17categories_RC):
        DATASET_NAME = category

    return MMLU_Category


def create_mmlu_tasks_withsplits(subject):
    class MMLU(GenericMMLU_withsplits):
        DATASET_NAME = subject

    return MMLU


# ---------------------------------------------------------------------------
# Merged variants: pruning + finetuning use the same merged data.
# Per-subject merged: concat the dev "validation" split (5 examples) with the
#   60% train portion of the shuffled HF test set.
# Per-category merged: concat the per-subject dev sets with the per-subject
#   60% train portions (already concatenated by _load_and_split_subjects).
# In both cases the dev rows would otherwise be tacked onto the end of the
# concatenated set, so we shuffle the merged set to interleave them.
# Test split (last 40% of shuffled HF test) is unchanged.
# ---------------------------------------------------------------------------


class GenericMMLU_Merged_withsplits(GenericMMLU_withsplits):
    """Per-subject MMLU merged variant: pruning and finetuning use the same data."""

    def training_docs(self):
        tot_test_size = len(self.dataset["test"])
        train_split = (
            self.dataset["test"]
            .shuffle(seed=0)
            .select(range(0, int(tot_test_size * self.TEST_FRACTION)))
        )
        merged = concatenate_datasets([train_split, self.dataset["validation"]]).shuffle(seed=0)
        return merged.map(self._process_doc, with_indices=True)

    def validation_docs(self):
        return self.training_docs()


def create_mmlu_merged_tasks_withsplits(subject):
    class MMLU_Merged(GenericMMLU_Merged_withsplits):
        DATASET_NAME = subject

    return MMLU_Merged


class MMLU_17categories_Merged_RC(MMLU_17categories_RC):
    """17-category MMLU merged variant: pruning and finetuning use the same data."""

    def training_docs(self):
        merged = concatenate_datasets([self.dataset["train"], self.dataset["validation"]]).shuffle(
            seed=0
        )
        return merged.map(self._process_doc, with_indices=True)

    def validation_docs(self):
        return self.training_docs()


def create_mmlu_categories_merged_tasks_withsplits(category):
    class MMLU_Category_Merged(MMLU_17categories_Merged_RC):
        DATASET_NAME = category

    return MMLU_Category_Merged


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


class MMLU_16clusters_RC(_MMLU_PerSubjectContext_RC):
    """MMLU task that groups subjects by router-based clustering (16 clusters)."""

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        _load_and_split_subjects(self, MMLU_CLUSTER_CATEGORIES, data_dir, cache_dir, download_mode)


def create_mmlu_cluster_tasks_withsplits(cluster_name):
    class MMLU_Cluster(MMLU_16clusters_RC):
        DATASET_NAME = cluster_name

    return MMLU_Cluster


# ---------------------------------------------------------------------------
# Router-clustering categories from k=16 spherical_kmeans on doc-level router
# embeddings (mean_pca_l2, balance off). Three variants by which router
# layers' probabilities are used:
#   _L0:   layer 0 only (127d)         — early-layer routing
#   _L15:  layer 15 only (127d)        — last-layer routing
#   _ALL:  all 16 layers (2032d)       — full-stack routing
#
# Model: twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32 step 30995
# Each subject is assigned to its dominant cluster (majority of its docs).
# Cluster names are auto-generated from each cluster's top-1 subject; the
# `_2` suffix on duplicates indicates the source clustering produced two
# distinct clusters whose top subjects strip to the same short name.
# Per-subject distributions saved alongside each clustering output:
#   claude_outputs/clustering/mmlu/<model>/<emb>_mean_pca_l2_spherical_kmeans_k16/
#       subject_distribution.{json,md}
# ---------------------------------------------------------------------------

MMLU_CLUSTER_CATEGORIES_L0 = {
    "cluster_l0_government_politics": [
        "global_facts",
        "high_school_government_and_politics",
        "international_law",
        "public_relations",
        "us_foreign_policy",
    ],
    "cluster_l0_biology": [
        "anatomy",
        "astronomy",
        "college_biology",
        "college_physics",
        "conceptual_physics",
        "high_school_biology",
        "high_school_chemistry",
        "medical_genetics",
        "virology",
    ],
    "cluster_l0_miscellaneous": [
        "miscellaneous",
    ],
    "cluster_l0_world_history": [
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "jurisprudence",
    ],
    "cluster_l0_moral_scenarios": [
        "moral_scenarios",
    ],
    "cluster_l0_law": [
        "professional_law",
    ],
    "cluster_l0_mathematics": [
        "elementary_mathematics",
        "high_school_computer_science",
    ],
    "cluster_l0_psychology": [
        "business_ethics",
        "high_school_psychology",
        "management",
    ],
    "cluster_l0_moral_disputes": [
        "moral_disputes",
        "philosophy",
        "security_studies",
        "sociology",
    ],
    "cluster_l0_psychology_2": [
        "human_aging",
        "professional_psychology",
    ],
    "cluster_l0_mathematics_2": [
        "abstract_algebra",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "econometrics",
        "electrical_engineering",
        "formal_logic",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "cluster_l0_accounting": [
        "professional_accounting",
    ],
    "cluster_l0_nutrition": [
        "clinical_knowledge",
        "college_medicine",
        "human_sexuality",
        "nutrition",
        "professional_medicine",
    ],
    "cluster_l0_prehistory": [
        "prehistory",
        "world_religions",
    ],
    "cluster_l0_marketing": [
        "computer_security",
        "logical_fallacies",
        "marketing",
    ],
    "cluster_l0_macroeconomics": [
        "high_school_geography",
        "high_school_macroeconomics",
        "high_school_microeconomics",
    ],
}


MMLU_CLUSTER_CATEGORIES_L15 = {
    "cluster_l15_security_studies": [
        "global_facts",
        "security_studies",
        "us_foreign_policy",
    ],
    "cluster_l15_mathematics": [
        "abstract_algebra",
        "college_computer_science",
        "college_mathematics",
        "elementary_mathematics",
        "formal_logic",
        "high_school_computer_science",
        "high_school_mathematics",
    ],
    "cluster_l15_marketing": [
        "business_ethics",
        "management",
        "marketing",
        "public_relations",
    ],
    "cluster_l15_biology": [
        "anatomy",
        "clinical_knowledge",
        "college_biology",
        "college_medicine",
        "high_school_biology",
        "human_sexuality",
        "medical_genetics",
        "nutrition",
        "professional_medicine",
        "virology",
    ],
    "cluster_l15_law": [
        "international_law",
        "jurisprudence",
        "professional_accounting",
        "professional_law",
    ],
    "cluster_l15_moral_disputes": [
        "logical_fallacies",
        "moral_disputes",
        "philosophy",
        "sociology",
    ],
    "cluster_l15_moral_scenarios": [
        "computer_security",
        "moral_scenarios",
    ],
    "cluster_l15_psychology": [
        "high_school_geography",
        "high_school_psychology",
        "human_aging",
    ],
    "cluster_l15_macroeconomics": [
        "high_school_macroeconomics",
        "high_school_microeconomics",
    ],
    "cluster_l15_chemistry": [
        "college_chemistry",
        "high_school_chemistry",
    ],
    "cluster_l15_miscellaneous": [
        "miscellaneous",
    ],
    "cluster_l15_conceptual_physics": [
        "astronomy",
        "college_physics",
        "conceptual_physics",
        "electrical_engineering",
        "high_school_physics",
    ],
    "cluster_l15_psychology_2": [
        "econometrics",
        "high_school_statistics",
        "machine_learning",
        "professional_psychology",
    ],
    "cluster_l15_prehistory": [
        "prehistory",
    ],
    "cluster_l15_government_politics": [
        "high_school_government_and_politics",
    ],
    "cluster_l15_world_history": [
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "world_religions",
    ],
}


MMLU_CLUSTER_CATEGORIES_ALL = {
    "cluster_all_nutrition": [
        "anatomy",
        "clinical_knowledge",
        "human_sexuality",
        "nutrition",
        "professional_medicine",
    ],
    "cluster_all_marketing": [
        "business_ethics",
        "computer_security",
        "management",
        "marketing",
        "public_relations",
    ],
    "cluster_all_law": [
        "international_law",
        "professional_law",
    ],
    "cluster_all_prehistory": [
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "prehistory",
        "world_religions",
    ],
    "cluster_all_mathematics": [
        "abstract_algebra",
        "college_computer_science",
        "college_mathematics",
        "econometrics",
        "elementary_mathematics",
        "formal_logic",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_statistics",
        "machine_learning",
    ],
    "cluster_all_moral_scenarios": [
        "moral_scenarios",
    ],
    "cluster_all_miscellaneous": [
        "miscellaneous",
    ],
    "cluster_all_biology": [
        "college_biology",
        "college_medicine",
        "high_school_biology",
        "medical_genetics",
        "virology",
    ],
    "cluster_all_security_studies": [
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
    "cluster_all_psychology": [
        "high_school_psychology",
        "human_aging",
        "professional_psychology",
    ],
    "cluster_all_chemistry": [
        "college_chemistry",
        "high_school_chemistry",
    ],
    "cluster_all_geography": [
        "global_facts",
        "high_school_geography",
        "high_school_government_and_politics",
    ],
    "cluster_all_macroeconomics": [
        "high_school_macroeconomics",
        "high_school_microeconomics",
    ],
    "cluster_all_moral_disputes": [
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "philosophy",
    ],
    "cluster_all_accounting": [
        "professional_accounting",
    ],
    "cluster_all_conceptual_physics": [
        "astronomy",
        "college_physics",
        "conceptual_physics",
        "electrical_engineering",
        "high_school_physics",
    ],
}


class MMLU_16clusters_L0_RC(_MMLU_PerSubjectContext_RC):
    """MMLU task grouped by router clustering on layer-0 probs (16 clusters)."""

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        _load_and_split_subjects(
            self, MMLU_CLUSTER_CATEGORIES_L0, data_dir, cache_dir, download_mode
        )


class MMLU_16clusters_L15_RC(_MMLU_PerSubjectContext_RC):
    """MMLU task grouped by router clustering on layer-15 probs (16 clusters)."""

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        _load_and_split_subjects(
            self, MMLU_CLUSTER_CATEGORIES_L15, data_dir, cache_dir, download_mode
        )


class MMLU_16clusters_ALL_RC(_MMLU_PerSubjectContext_RC):
    """MMLU task grouped by router clustering on all-layer probs (16 clusters)."""

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        _load_and_split_subjects(
            self, MMLU_CLUSTER_CATEGORIES_ALL, data_dir, cache_dir, download_mode
        )


def create_mmlu_cluster_l0_tasks_withsplits(cluster_name):
    class MMLU_Cluster_L0(MMLU_16clusters_L0_RC):
        DATASET_NAME = cluster_name

    return MMLU_Cluster_L0


def create_mmlu_cluster_l15_tasks_withsplits(cluster_name):
    class MMLU_Cluster_L15(MMLU_16clusters_L15_RC):
        DATASET_NAME = cluster_name

    return MMLU_Cluster_L15


def create_mmlu_cluster_all_tasks_withsplits(cluster_name):
    class MMLU_Cluster_ALL(MMLU_16clusters_ALL_RC):
        DATASET_NAME = cluster_name

    return MMLU_Cluster_ALL
