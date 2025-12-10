import shutil
from pathlib import Path
from olmo_core.nn.transformer import TransformerConfig
from scripts.akshitab.add_finegrained_expert.add_new_expert import save_checkpoint

if __name__ == "__main__":
    co = TransformerConfig.smallmoe(100, n_layers=2, d_model=128)
    mo = co.build()
    output_path = Path("src/test_fixtures/smallmoe")
    if output_path.exists():
        shutil.rmtree(output_path)
    save_checkpoint({"model": co.as_config_dict()}, mo, str(output_path))
