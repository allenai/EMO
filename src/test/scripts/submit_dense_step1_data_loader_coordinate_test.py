import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_submitter_module():
    path = (
        Path(__file__).parents[3]
        / "scripts"
        / "models"
        / "submit_dense_step1_data_loader_coordinate.py"
    )
    spec = importlib.util.spec_from_file_location("dense_data_loader_submitter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_embedding_decay_has_a_distinct_canonical_trajectory():
    module = _load_submitter_module()
    base = dict(
        method="dynamic_repacking",
        global_sequences=512,
        learning_rate="1e-3",
        weight_decay="1.0",
        weight_tying=True,
    )

    zero_embedding_wd = module.trajectory_output(
        SimpleNamespace(**base, decay_embeddings=False)
    )
    global_embedding_wd = module.trajectory_output(
        SimpleNamespace(**base, decay_embeddings=True)
    )

    assert zero_embedding_wd.endswith("/bs512_dr_wt_lr1e-3_wd1.0")
    assert global_embedding_wd.endswith("/bs512_dr_wt_embwd_lr1e-3_wd1.0")
    assert module.method_key("dynamic_repacking", 512, True, False) == "drwt512"
    assert module.method_key("dynamic_repacking", 512, True, True) == "drwtembwd512"
