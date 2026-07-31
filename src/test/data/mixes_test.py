import pytest

from olmo_core.data import DataMix, TokenizerName
from olmo_core.io import file_exists


def test_olmoe_mix():
    from botocore.exceptions import NoCredentialsError

    paths, labels = DataMix.OLMoE_mix_0824.build("s3://ai2-llm", TokenizerName.dolma2)
    assert len(paths) == len(labels)
    assert (
        paths[-1]
        == "s3://ai2-llm/preprocessed/olmo-mix/danyh-compiled-v1_7/documents/wiki/allenai/dolma2-tokenizer/part-1-00000.npy"
    )

    try:
        assert file_exists(paths[-1])
    except NoCredentialsError:
        pytest.skip("Requires AWS credentials")


@pytest.mark.parametrize(
    ("mix", "tokenizer", "expected_count", "first_suffix", "last_suffix"),
    [
        (
            DataMix.dclm_120B,
            TokenizerName.dolma2,
            30,
            "allenai/dolma2-tokenizer/part-000-00000.npy",
            "allenai/dolma2-tokenizer/part-005-00004.npy",
        ),
        (
            DataMix.dclm_700B,
            TokenizerName.dolma2_sigdig,
            192,
            "dolma2-tokenizer-sigdig/tokenizer.json_bos/part-00-00000.npy",
            "dolma2-tokenizer-sigdig/tokenizer.json_bos/part-95-00001.npy",
        ),
        (
            DataMix.dclm_full,
            TokenizerName.dolma2,
            940,
            "allenai/dolma2-tokenizer/part-000-00000.npy",
            "allenai/dolma2-tokenizer/part-187-00004.npy",
        ),
    ],
)
def test_dclm_data_only_mixes(mix, tokenizer, expected_count, first_suffix, last_suffix):
    assert DataMix(mix.value) is mix
    paths, labels = mix.build("s3://ai2-llm", tokenizer)

    assert len(paths) == expected_count
    assert len(paths) == len(set(paths))
    assert labels == ["dclm"] * expected_count
    assert paths[0].endswith(first_suffix)
    assert paths[-1].endswith(last_suffix)


def test_dclm_700b_rejects_unavailable_tokenizer():
    with pytest.raises(ValueError, match="only available with the dolma2 sigdig tokenizer"):
        DataMix.dclm_700B.build("s3://ai2-llm", TokenizerName.dolma2)


def test_dolma17_mix():
    from botocore.exceptions import NoCredentialsError

    paths, labels = DataMix.dolma17.build("s3://ai2-llm", TokenizerName.gpt_neox_olmo_dolma_v1_5)
    assert len(paths) == len(labels)
    assert (
        paths[-1]
        == "s3://ai2-llm/preprocessed/olmo-mix/v1_7-dd_ngram_dp_030-qc_cc_en_bin_001/cc_en_tail/gpt-neox-olmo-dolma-v1_5/part-092-00000.npy"
    )

    try:
        assert file_exists(paths[-1])
    except NoCredentialsError:
        pytest.skip("Requires AWS credentials")


def test_v3_small_ppl_validation_mix():
    from botocore.exceptions import NoCredentialsError

    paths, labels = DataMix.v3_small_ppl_validation.build("s3://ai2-llm", TokenizerName.dolma2)
    assert len(paths) == len(labels)
    assert (
        paths[0]
        == "s3://ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/c4_en/val/part-0-00000.npy"
    )
    assert labels[0] == "c4_en-validation"

    try:
        assert file_exists(paths[-1])
    except NoCredentialsError:
        pytest.skip("Requires AWS credentials")


def test_code_fresh_ppl_validation_mix():
    paths, labels = DataMix.code_fresh_ppl_validation.build(
        "s3://ai2-llm-public", TokenizerName.dolma2
    )
    assert len(paths) == len(labels)
    assert (
        paths[0]
        == "s3://ai2-llm-public/eval-data/perplexity/code_fresh_0825_1225_dolma2-tokenizer/blade/val/part-0-00000.npy"
    )
    assert labels[0] == "code_fresh_blade-validation"
