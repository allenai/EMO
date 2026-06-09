#!/usr/bin/env python3
"""
Convert an EMO OLMo-core checkpoint to a HuggingFace checkpoint using stock
transformers — no custom fork required.

``olmo_core.nn.hf.config`` expects ``EmoConfig`` / ``EmoSharedConfig`` /
``Olmo2NoQKNormPrenormConfig`` to be importable from ``transformers``; those
classes only exist natively in the ``ryanyxw/transformers#flexmoe_v4_57_1``
fork. This driver injects the equivalent classes from
``src/hf_trust_remote_code/`` (the same trust_remote_code files end users load
the released checkpoints with) into the ``transformers`` namespace and
registers them with the ``Auto*`` factories, then delegates to the standard
conversion entry point. The registration is in-process only — the installed
transformers library is never modified.

After conversion, the remote-code files (``modeling_emo.py`` etc.) are staged
into the output dir and ``auto_map`` is patched into ``config.json`` — via
``stage_remote_code()`` from ``src/hf_trust_remote_code/upload_to_hf.py`` — so
the checkpoint is directly loadable with ``trust_remote_code=True``. Pass
``--no-stage`` to skip this and get a bare converted checkpoint.

Usage mirrors ``src/examples/huggingface/convert_checkpoint_to_hf.py``::

    python scripts/convert_emo_to_hf.py -i /path/to/step30995 -o /path/to/step30995-hf
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRC_DIR = REPO_ROOT / "src" / "hf_trust_remote_code"


def register_local_hf_classes() -> None:
    """Make the trust_remote_code classes importable from ``transformers``.

    Must run before any ``olmo_core.nn.hf`` import — that module resolves
    ``EmoConfig`` etc. from the ``transformers`` namespace at import time.
    """
    import importlib
    import types

    def _trc_module(pkg_name: str, dirname: str, module: str):
        # The trust_remote_code files use relative imports (they're written for
        # HF's dynamic-module loader), so expose each folder as a synthetic
        # package rather than putting it on sys.path directly.
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(TRC_DIR / dirname)]
            sys.modules[pkg_name] = pkg
        return importlib.import_module(f"{pkg_name}.{module}")

    EmoConfig = _trc_module("hf_trc_emo", "emo", "configuration_emo").EmoConfig
    EmoForCausalLM = _trc_module("hf_trc_emo", "emo", "modeling_emo").EmoForCausalLM
    Olmo2NoQKNormPrenormConfig = _trc_module(
        "hf_trc_olmo2", "olmo2_noqknorm_prenorm", "configuration_olmo2_noqknorm_prenorm"
    ).Olmo2NoQKNormPrenormConfig
    Olmo2NoQKNormPrenormForCausalLM = _trc_module(
        "hf_trc_olmo2", "olmo2_noqknorm_prenorm", "modeling_olmo2_noqknorm_prenorm"
    ).Olmo2NoQKNormPrenormForCausalLM

    # Import transformers only AFTER the trust_remote_code modules: their import
    # chain replaces the `transformers` entry in sys.modules with a fresh lazy
    # module, so attributes set on the old object would be lost.
    import transformers

    class EmoSharedConfig(transformers.PretrainedConfig):
        """Placeholder so ``from transformers import EmoConfig, EmoSharedConfig``
        succeeds. Shared-MLP checkpoints have no trust_remote_code
        implementation and still require the fork."""

        model_type = "emo_shared"

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "EmoSharedConfig (shared-MLP variant) is not supported by this "
                "fork-free driver; install ryanyxw/transformers#flexmoe_v4_57_1."
            )

    transformers.EmoConfig = EmoConfig
    transformers.EmoSharedConfig = EmoSharedConfig
    transformers.Olmo2NoQKNormPrenormConfig = Olmo2NoQKNormPrenormConfig

    transformers.AutoConfig.register("emo", EmoConfig)
    transformers.AutoModelForCausalLM.register(EmoConfig, EmoForCausalLM)
    transformers.AutoConfig.register("olmo2_noqknorm_prenorm", Olmo2NoQKNormPrenormConfig)
    transformers.AutoModelForCausalLM.register(
        Olmo2NoQKNormPrenormConfig, Olmo2NoQKNormPrenormForCausalLM
    )


def parse_args():
    import torch

    from olmo_core.config import DType

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--checkpoint-input-path", type=str, required=True)
    parser.add_argument("-o", "--huggingface-output-dir", type=str, required=True)
    parser.add_argument("-s", "--max-sequence-length", type=int)
    parser.add_argument("-t", "--tokenizer")
    parser.add_argument("--skip-validation", dest="validate", action="store_false")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=torch.device)
    parser.add_argument("--dtype", type=DType, default=DType.float32)
    parser.add_argument("--validation-device", type=torch.device)
    parser.add_argument("--validation-sliding-window", type=int)
    parser.add_argument("--moe-capacity-factor", type=float)
    parser.add_argument(
        "--no-stage",
        dest="stage",
        action="store_false",
        help="Skip staging the trust_remote_code files into the output dir.",
    )
    return parser.parse_args()


def stage_trust_remote_code(output_dir: str) -> None:
    """Copy the matching trust_remote_code files into the converted checkpoint
    and patch ``auto_map``, reusing the helpers from upload_to_hf.py."""
    import importlib.util

    out = Path(output_dir)
    if not (out / "config.json").is_file():
        print(f"NOTE: skipping trust_remote_code staging — {out} is not a local HF checkpoint dir")
        return

    spec = importlib.util.spec_from_file_location("upload_to_hf", TRC_DIR / "upload_to_hf.py")
    assert spec is not None and spec.loader is not None
    upload_to_hf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upload_to_hf)

    arch = upload_to_hf.detect_architecture(out / "config.json")
    print(f"Staging trust_remote_code files for {arch} into {out}")
    upload_to_hf.stage_remote_code(out, arch)


def main():
    register_local_hf_classes()

    # olmo_core.nn.hf must be imported after registration (see above).
    from olmo_core.nn.hf import convert_checkpoint_to_hf, load_config
    from olmo_core.utils import prepare_cli_environment

    prepare_cli_environment()
    args = parse_args()

    experiment_config = load_config(args.checkpoint_input_path)
    if experiment_config is None:
        raise RuntimeError("Experiment config not found, cannot convert to HF checkpoint")

    transformer_config_dict = experiment_config["model"]
    tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer")

    assert transformer_config_dict is not None
    assert tokenizer_config_dict is not None

    # Models trained with eval_document_expert_pool < num_experts mask out
    # low-scoring experts per document at eval time, but the HF EmoForCausalLM
    # (the released inference semantics) always routes over all experts. Set
    # the pool to cover all experts so validation compares pure weight
    # conversion rather than this intentional behavior difference. This is a
    # runtime routing knob — it does not affect the converted weights.
    moe = transformer_config_dict.get("block", {}).get("feed_forward_moe") or {}
    router = moe.get("router") or {}
    if "eval_document_expert_pool" in router:
        num_experts = moe["num_experts"]
        if router["eval_document_expert_pool"] != num_experts:
            print(
                f"NOTE: overriding eval_document_expert_pool "
                f"{router['eval_document_expert_pool']} -> {num_experts} so validation "
                f"routes over all experts, matching HF inference semantics"
            )
            router["eval_document_expert_pool"] = num_experts

    convert_checkpoint_to_hf(
        original_checkpoint_path=args.checkpoint_input_path,
        output_path=args.huggingface_output_dir,
        transformer_config_dict=transformer_config_dict,
        tokenizer_config_dict=tokenizer_config_dict,
        dtype=args.dtype,
        max_sequence_length=args.max_sequence_length,
        tokenizer_id=args.tokenizer,
        validate=args.validate,
        debug=args.debug,
        device=args.device,
        moe_capacity_factor=args.moe_capacity_factor,
        validation_device=args.validation_device or args.device,
        validation_sliding_window=args.validation_sliding_window,
    )

    if args.stage:
        stage_trust_remote_code(args.huggingface_output_dir)


if __name__ == "__main__":
    main()
