# this is specifically for replicating figure 4 of olmoe paper
pip install -e ".[eval]"
# Override transformers and vllm versions, bypassing pyproject conflicts with olmes.
# TODO: remove once merged
pip install vllm==0.7.0; pip uninstall -y transformers; pip install "transformers@git+https://github.com/ryanyxw/olmoefig4-flexolmo_transformers"; pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124; pip install ipdb;