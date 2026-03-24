from typing import Dict

from oe_eval.data.mmlu_pro_categories import MMLU_PRO_CATEGORIES
from oe_eval.data.mmlu_tasks import MMLU_SUBJECTS
from oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY
from oe_eval.tasks.oe_eval_tasks.mmlu import create_mmlu_task
from oe_eval.tasks.oe_eval_tasks.mmlu_pro import create_mmlu_pro_task

from .tasks.splits_mmlu import (
    create_mmlu_tasks_withsplits,
    create_mmlu_categories_tasks_withsplits,
    create_mmlu_cluster_tasks_withsplits,
    MMLU_CLUSTER_CATEGORIES,
)
from .tasks.splits_mmlu_pro import (
    create_mmlu_pro_category_tasks_withsplits,
    create_mmlu_pro_merged_tasks_withsplits,
    MMLU_PRO_CATEGORIES_MAP,
)
from .tasks import (
    agi_eval,
    hatespeech,
    medmcqa,
    medqa,
    narrativeqa,
    news_gen,
    poem_gen,
    ruler,
    sciriff,
    splits_arc,
    splits_boolq,
    splits_coqa,
    splits_csqa,
    splits_gsm8k,
    splits_hellaswag,
    splits_openbookqa,
    splits_piqa,
    splits_siqa,
    splits_squad,
    splits_synthea,
    splits_winogrande,
    squad,
    squad2,
    story_gen,
    xsum,
)


def create_core_mmlu_tasks_withsplits():
    """Creates a dictionary of tasks from a list of subjects.
    Note that the differences between train, validation, and test is declared in TASK_CONFIGS"""
    res = {}
    for sub in MMLU_SUBJECTS:
        res[f"mmlu_{sub}:rc_validation"] = create_mmlu_tasks_withsplits(sub)
        res[f"mmlu_{sub}:rc_test"] = create_mmlu_tasks_withsplits(sub)
        res[f"mmlu_{sub}:rc_train"] = create_mmlu_tasks_withsplits(sub)
    return res

def create_category_mmlu_tasks_withsplits():
    """Creates a dictionary of tasks from a list of subjects.
    Note that the differences between train, validation, and test is declared in TASK_CONFIGS"""
    MMLU_CATEGORIES = [
        "biology",
        "business",
        "chemistry",
        "computer_science",
        "culture",
        "economics",
        "engineering",
        "geography",
        "health",
        "history",
        "law",
        "math",
        "other",
        "philosophy_cat",
        "physics",
        "politics",
        "psychology",
    ]
    res = {}
    for sub in MMLU_CATEGORIES:
        res[f"mmlu_{sub}:rc_validation"] = create_mmlu_categories_tasks_withsplits(sub)
        res[f"mmlu_{sub}:rc_test"] = create_mmlu_categories_tasks_withsplits(sub)
        res[f"mmlu_{sub}:rc_train"] = create_mmlu_categories_tasks_withsplits(sub)
    return res


def create_cluster_mmlu_tasks_withsplits():
    """Creates MMLU tasks grouped by router-based clustering (16 clusters)."""
    res = {}
    for cluster_name in MMLU_CLUSTER_CATEGORIES:
        res[f"mmlu_{cluster_name}:rc_validation"] = create_mmlu_cluster_tasks_withsplits(cluster_name)
        res[f"mmlu_{cluster_name}:rc_test"] = create_mmlu_cluster_tasks_withsplits(cluster_name)
        res[f"mmlu_{cluster_name}:rc_train"] = create_mmlu_cluster_tasks_withsplits(cluster_name)
    return res


def create_core_mmlu_pro_tasks_withsplits():
    """Creates a dictionary of MMLU-Pro tasks from a list of categories
    Note that the differences between train, validation, and test is declared in TASK_CONFIGS"""
    res = {}
    for sub in MMLU_PRO_CATEGORIES:
        res[f"mmlu_pro_{sub}:mc_validation"] = create_mmlu_pro_task(sub)
        res[f"mmlu_pro_{sub}:mc_test"] = create_mmlu_pro_task(sub)
        res[f"mmlu_pro_{sub}:rc_validation"] = create_mmlu_pro_task(sub, is_mc=False)
        res[f"mmlu_pro_{sub}:rc_test"] = create_mmlu_pro_task(sub, is_mc=False)
    return res


def create_category_mmlu_pro_tasks_withsplits():
    """Creates MMLU-Pro tasks with custom train/validation/test splits for the pruning pipeline."""
    res = {}
    for cat in MMLU_PRO_CATEGORIES_MAP:
        res[f"mmlu_pro_{cat}:rc_validation"] = create_mmlu_pro_category_tasks_withsplits(cat)
        res[f"mmlu_pro_{cat}:rc_test"] = create_mmlu_pro_category_tasks_withsplits(cat)
        res[f"mmlu_pro_{cat}:rc_train"] = create_mmlu_pro_category_tasks_withsplits(cat)
    return res


def create_merged_mmlu_pro_tasks_withsplits():
    """Creates MMLU-Pro merged variant: pruning and finetuning use the same combined data."""
    res = {}
    for cat in MMLU_PRO_CATEGORIES_MAP:
        res[f"mmlu_pro_merged_{cat}:rc_validation"] = create_mmlu_pro_merged_tasks_withsplits(cat)
        res[f"mmlu_pro_merged_{cat}:rc_test"] = create_mmlu_pro_merged_tasks_withsplits(cat)
        res[f"mmlu_pro_merged_{cat}:rc_train"] = create_mmlu_pro_merged_tasks_withsplits(cat)
    return res


new_task_registry: Dict = {
    "xsum": xsum.XSum,
    "narrativeqa": narrativeqa.NarrativeQA,
    "story_gen": story_gen.Story_Gen_LMJudge,
    "news_gen": news_gen.News_Gen_LMJudge,
    "poem_gen": poem_gen.Poem_Gen_LMJudge,
    "medqa": medqa.MedQA,
    "medmcqa:mc": medmcqa.MedMCQAMC,
    "hatespeech18": hatespeech.HateSpeech18,
    "tweet_eval_hate": hatespeech.TweetEvalHate,
    "hate_speech_offensive": hatespeech.HateSpeechOffensive,
    "hatexplain": hatespeech.Hatexplain,
    **agi_eval.create_core_agi_eval_tasks(),
    **ruler.create_ruler_tasks(),
    **sciriff.create_bio_sciriff_qa_tasks(),
    "squad": squad.SQuAD,
    "squad2": squad2.SQuAD2,
    "arc_easy:mc_train": splits_arc.ARCEasy_MC_Train,
    "arc_easy:mc_validation": splits_arc.ARCEasy_MC_Validation,
    "arc_easy:mc_test": splits_arc.ARCEasy_MC_Test,
    "arc_easy:rc_train": splits_arc.ARCEasy_RC_Base,
    "arc_easy:rc_validation": splits_arc.ARCEasy_RC_Base,
    "arc_easy:rc_train_0shot": splits_arc.ARCEasy_RC_Base,
    "arc_easy:rc_validation_0shot": splits_arc.ARCEasy_RC_Base,
    "arc_easy:rc_test": splits_arc.ARCEasy_RC_Base,
    "arc_challenge:mc_train": splits_arc.ARCChallenge_MC_Train,
    "arc_challenge:mc_validation": splits_arc.ARCChallenge_MC_Validation,
    "arc_challenge:mc_test": splits_arc.ARCChallenge_MC_Test,
    "arc_challenge:rc_train": splits_arc.ARCChallenge_RC_Base,
    "arc_challenge:rc_validation": splits_arc.ARCChallenge_RC_Base,
    "arc_challenge:rc_train_0shot": splits_arc.ARCChallenge_RC_Base,
    "arc_challenge:rc_validation_0shot": splits_arc.ARCChallenge_RC_Base,
    "arc_challenge:rc_test": splits_arc.ARCChallenge_RC_Base,
    "boolq:mc_train": splits_boolq.BoolQ_MC_Train,
    "boolq:mc_validation": splits_boolq.BoolQ_MC_Validation,
    "boolq:mc_test": splits_boolq.BoolQ_MC_Test,
    "boolq:rc_train": splits_boolq.BoolQ_RC_Base,
    "boolq:rc_validation": splits_boolq.BoolQ_RC_Base,
    "boolq:rc_train_0shot": splits_boolq.BoolQ_RC_Base,
    "boolq:rc_validation_0shot": splits_boolq.BoolQ_RC_Base,
    "boolq:rc_test": splits_boolq.BoolQ_RC_Base,
    "csqa:mc_train": splits_csqa.CommonsenseQA_MC_Train,
    "csqa:mc_validation": splits_csqa.CommonsenseQA_MC_Validation,
    "csqa:mc_test": splits_csqa.CommonsenseQA_MC_Test,
    "csqa:rc_train": splits_csqa.CommonsenseQA_RC_Base,
    "csqa:rc_validation": splits_csqa.CommonsenseQA_RC_Base,
    "csqa:rc_train_0shot": splits_csqa.CommonsenseQA_RC_Base,
    "csqa:rc_validation_0shot": splits_csqa.CommonsenseQA_RC_Base,
    "csqa:rc_test": splits_csqa.CommonsenseQA_RC_Base,
    "hellaswag:mc_train": splits_hellaswag.HellaSwag_MC_Train,
    "hellaswag:mc_validation": splits_hellaswag.HellaSwag_MC_Validation,
    "hellaswag:mc_test": splits_hellaswag.HellaSwag_MC_Test,
    "hellaswag:rc_train": splits_hellaswag.HellaSwag_RC_Base,
    "hellaswag:rc_validation": splits_hellaswag.HellaSwag_RC_Base,
    "hellaswag:rc_train_0shot": splits_hellaswag.HellaSwag_RC_Base,
    "hellaswag:rc_validation_0shot": splits_hellaswag.HellaSwag_RC_Base,
    "hellaswag:rc_test": splits_hellaswag.HellaSwag_RC_Base,
    "openbookqa:mc_train": splits_openbookqa.OpenBookQA_MC_Train,
    "openbookqa:mc_validation": splits_openbookqa.OpenBookQA_MC_Validation,
    "openbookqa:mc_test": splits_openbookqa.OpenBookQA_MC_Test,
    "openbookqa:rc_train": splits_openbookqa.OpenBookQA_RC_Base,
    "openbookqa:rc_validation": splits_openbookqa.OpenBookQA_RC_Base,
    "openbookqa:rc_train_0shot": splits_openbookqa.OpenBookQA_RC_Base,
    "openbookqa:rc_validation_0shot": splits_openbookqa.OpenBookQA_RC_Base,
    "openbookqa:rc_test": splits_openbookqa.OpenBookQA_RC_Base,
    "piqa:mc_train": splits_piqa.PIQA_MC_Train,
    "piqa:mc_validation": splits_piqa.PIQA_MC_Validation,
    "piqa:mc_test": splits_piqa.PIQA_MC_Test,
    "piqa:rc_train": splits_piqa.PIQA_RC_Base,
    "piqa:rc_validation": splits_piqa.PIQA_RC_Base,
    "piqa:rc_train_0shot": splits_piqa.PIQA_RC_Base,
    "piqa:rc_validation_0shot": splits_piqa.PIQA_RC_Base,
    "piqa:rc_test": splits_piqa.PIQA_RC_Base,
    "socialiqa:mc_train": splits_siqa.SocialIQA_MC_Train,
    "socialiqa:mc_validation": splits_siqa.SocialIQA_MC_Validation,
    "socialiqa:mc_test": splits_siqa.SocialIQA_MC_Test,
    "socialiqa:rc_train": splits_siqa.SocialIQA_RC_Base,
    "socialiqa:rc_validation": splits_siqa.SocialIQA_RC_Base,
    "socialiqa:rc_train_0shot": splits_siqa.SocialIQA_RC_Base,
    "socialiqa:rc_validation_0shot": splits_siqa.SocialIQA_RC_Base,
    "socialiqa:rc_test": splits_siqa.SocialIQA_RC_Base,
    "winogrande:mc_train": splits_winogrande.Winogrande_MC_Train,
    "winogrande:mc_validation": splits_winogrande.Winogrande_MC_Validation,
    "winogrande:mc_test": splits_winogrande.Winogrande_MC_Test,
    "winogrande:rc_train": splits_winogrande.Winogrande_RC_Base,
    "winogrande:rc_validation": splits_winogrande.Winogrande_RC_Base,
    "winogrande:rc_train_0shot": splits_winogrande.Winogrande_RC_Base,
    "winogrande:rc_validation_0shot": splits_winogrande.Winogrande_RC_Base,
    "winogrande:rc_test": splits_winogrande.Winogrande_RC_Base,
    "synthea:rc_train": splits_synthea.Synthea_RC_Train,
    "synthea:rc_validation": splits_synthea.Synthea_RC_Validation,
    "synthea:rc_test": splits_synthea.Synthea_RC_Test,
    "synthea:rc_train_0shot": splits_synthea.Synthea_RC_Train_0shot,
    "synthea:rc_validation_0shot": splits_synthea.Synthea_RC_Validation_0shot,
    "synthea:rc_test_0shot": splits_synthea.Synthea_RC_Test_0shot,
    # MMLU
    **create_core_mmlu_tasks_withsplits(),
    **create_category_mmlu_tasks_withsplits(),
    **create_cluster_mmlu_tasks_withsplits(),
    # **create_core_mmlu_pro_tasks_withsplits(),
    **create_category_mmlu_pro_tasks_withsplits(),
    **create_merged_mmlu_pro_tasks_withsplits(),
    # GSM8K
    "gsm8k_perplexity:train": splits_gsm8k.GSM8K_Perplexity_Base,
    "gsm8k_perplexity:validation": splits_gsm8k.GSM8K_Perplexity_Base,
    "gsm8k_perplexity:test": splits_gsm8k.GSM8K_Perplexity_Base,
    "gsm8k_perplexity_0shot:train": splits_gsm8k.GSM8K_Perplexity_Base,
    "gsm8k_perplexity_0shot:validation": splits_gsm8k.GSM8K_Perplexity_Base,
    "gsm8k_perplexity_0shot:test": splits_gsm8k.GSM8K_Perplexity_Base,
    "gsm8k_generation_0shot:train": splits_gsm8k.GSM8K_Generation_Train_0shot,
    "gsm8k_generation_0shot:validation": splits_gsm8k.GSM8K_Generation_Validation_0shot,
    "gsm8k_generation_0shot:test": splits_gsm8k.GSM8K_Generation_Test_0shot,
    "gsm8k_generation_8shot:train": splits_gsm8k.GSM8K_Generation_Train_8shot,
    "gsm8k_generation_8shot:validation": splits_gsm8k.GSM8K_Generation_Validation_8shot,
    "gsm8k_generation_8shot:test": splits_gsm8k.GSM8K_Generation_Test_8shot,
    "coqa_0shot:train": splits_coqa.COQA_Train_0shot,
    "coqa_0shot:validation": splits_coqa.COQA_Validation_0shot,
    "coqa_0shot:test": splits_coqa.COQA_Test_0shot,
    "coqa_full_0shot:train": splits_coqa.COQA_full_Train_0shot,
    "coqa_full_0shot:validation": splits_coqa.COQA_full_Validation_0shot,
    "coqa_full_0shot:test": splits_coqa.COQA_full_Test_0shot,
    "squad_0shot:train": splits_squad.SQUAD_Train_0shot,
    "squad_0shot:validation": splits_squad.SQUAD_Validation_0shot,
    "squad_0shot:test": splits_squad.SQUAD_Test_0shot,
}

TASK_REGISTRY.update(new_task_registry)
