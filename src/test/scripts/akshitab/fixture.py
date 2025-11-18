
from olmo_core.nn.transformer import TransformerConfig
from scripts.akshitab.add_finegrained_expert.add_new_expert import save_checkpoint

if __name__ == "__main__":
    co = TransformerConfig.smallmoe(1000)
    mo = co.build()
    save_checkpoint({"model": co.as_config_dict()}, mo, "src/test_fixtures/smallmoe")
