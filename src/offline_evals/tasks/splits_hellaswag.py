import json
import os

from datasets import concatenate_datasets
from oe_eval.tasks.oe_eval_tasks.hellaswag import HellaSwag, HellaSwagMC

from ..metrics.mc_softloss import SoftLoss

# Path to cluster assignments JSON (produced by assign_hellaswag_clusters.py).
# Set via HELLASWAG_CLUSTER_ASSIGNMENTS_FILE env var or defaults to this path.
_DEFAULT_CLUSTER_ASSIGNMENTS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "hellaswag_cluster_assignments.json",
)
HELLASWAG_CLUSTER_ASSIGNMENTS_FILE = os.environ.get(
    "HELLASWAG_CLUSTER_ASSIGNMENTS_FILE",
    os.path.normpath(_DEFAULT_CLUSTER_ASSIGNMENTS_FILE),
)

_cluster_assignments_cache = None

def _load_cluster_assignments():
    """Load and cache cluster assignments from JSON file."""
    global _cluster_assignments_cache
    if _cluster_assignments_cache is None:
        with open(HELLASWAG_CLUSTER_ASSIGNMENTS_FILE) as f:
            _cluster_assignments_cache = json.load(f)
    return _cluster_assignments_cache


class HellaSwag_RC_Base(HellaSwag):
    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
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

    def make_metrics(self):
        # run the super
        super().make_metrics()
        # add softloss metric
        self._metrics += [SoftLoss(**self.task_config["metric_kwargs"])]

        return self._metrics


class HellaSwag_MC_Base(HellaSwagMC):
    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
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


# ---------------------------------------------------------------------------
# Merged variant: train+val combined for both pruning and finetuning
# (similar to GenericMMLUPro_merged in splits_mmlu_pro.py)
# ---------------------------------------------------------------------------

class HellaSwag_Merged_RC(HellaSwag_RC_Base):
    """HellaSwag variant where pruning and finetuning use the same merged data.

    Merges train + validation into a single set used for both pipeline phases:
      - validation/train: merged set (38,905 + 1,000 = 39,905) for both pruning and finetuning
      - test: original HF validation split (10,042) for evaluation (unchanged)
    """

    def training_docs(self):
        if self._training_docs is None:
            train_dataset = (
                self.dataset["train"]
                .shuffle(seed=0)
                .select(range(1000, len(self.dataset["train"])))
            )
            val_dataset = (
                self.dataset["train"]
                .shuffle(seed=0)
                .select(range(0, 1000))
            )
            merged = concatenate_datasets([train_dataset, val_dataset])
            self._training_docs = list(map(self._process_doc, merged))
        return self._training_docs

    def validation_docs(self):
        # Same merged set as training_docs
        return self.training_docs()


# ---------------------------------------------------------------------------
# Cluster variants: per-cluster subsets of HellaSwag
# ---------------------------------------------------------------------------

def _get_split_cluster_indices(cluster_id, split):
    """Get cluster indices mapped to positions within each split's dataset.

    The JSON stores indices as positions in the train_val array (0-39904),
    where rows 0..N_train-1 are train and N_train..N_train+N_val-1 are validation.
    This function remaps validation indices from [N_train, N_train+N_val) to [0, N_val).
    Test indices are already positions within the test-only subset.
    """
    data = _load_cluster_assignments()
    raw_indices = data["clusters"][str(cluster_id)][split]
    if split == "validation":
        # Remap from train_val positions to validation-only positions
        # In train_val: train is [0, N_train), validation is [N_train, N_train+N_val)
        n_train = sum(
            len(data["clusters"][str(c)]["train"])
            for c in range(data["k"])
        )
        return [i - n_train for i in raw_indices]
    return raw_indices


class HellaSwag_Cluster_RC(HellaSwag_RC_Base):
    """HellaSwag filtered to a single cluster's examples.

    Uses cluster assignments from a JSON file to select only examples
    belonging to one cluster. The cluster_id is set via task_config.
    """
    CLUSTER_ID = None  # Set by factory function

    def training_docs(self):
        if self._training_docs is None:
            full_train = (
                self.dataset["train"]
                .shuffle(seed=0)
                .select(range(1000, len(self.dataset["train"])))
            )
            indices = _get_split_cluster_indices(self.CLUSTER_ID, "train")
            selected = full_train.select(indices)
            self._training_docs = list(map(self._process_doc, selected))
        return self._training_docs

    def validation_docs(self):
        full_val = (
            self.dataset["train"]
            .shuffle(seed=0)
            .select(range(0, 1000))
        )
        indices = _get_split_cluster_indices(self.CLUSTER_ID, "validation")
        selected = full_val.select(indices)
        return map(self._process_doc, selected)

    def test_docs(self):
        indices = _get_split_cluster_indices(self.CLUSTER_ID, "test")
        selected = self.dataset["validation"].select(indices)
        return map(self._process_doc, selected)


class HellaSwag_Cluster_Merged_RC(HellaSwag_RC_Base):
    """Per-cluster HellaSwag with train+val merged for both pruning and finetuning.

    Combines cluster filtering with merging:
      - validation/train: cluster's train + validation merged
      - test: cluster's test examples
    """
    CLUSTER_ID = None  # Set by factory function

    def training_docs(self):
        if self._training_docs is None:
            full_train = (
                self.dataset["train"]
                .shuffle(seed=0)
                .select(range(1000, len(self.dataset["train"])))
            )
            full_val = (
                self.dataset["train"]
                .shuffle(seed=0)
                .select(range(0, 1000))
            )
            train_indices = _get_split_cluster_indices(self.CLUSTER_ID, "train")
            val_indices = _get_split_cluster_indices(self.CLUSTER_ID, "validation")
            selected_train = full_train.select(train_indices)
            selected_val = full_val.select(val_indices)
            merged = concatenate_datasets([selected_train, selected_val])
            self._training_docs = list(map(self._process_doc, merged))
        return self._training_docs

    def validation_docs(self):
        # Same merged set as training_docs
        return self.training_docs()

    def test_docs(self):
        indices = _get_split_cluster_indices(self.CLUSTER_ID, "test")
        selected = self.dataset["validation"].select(indices)
        return map(self._process_doc, selected)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_hellaswag_cluster_task(cluster_id):
    class HS_Cluster(HellaSwag_Cluster_RC):
        CLUSTER_ID = cluster_id
    return HS_Cluster


def create_hellaswag_cluster_merged_task(cluster_id):
    class HS_ClusterMerged(HellaSwag_Cluster_Merged_RC):
        CLUSTER_ID = cluster_id
    return HS_ClusterMerged


NUM_HELLASWAG_CLUSTERS = 6
