from typing import Dict

from oe_eval.data.mmlu_pro_categories import MMLU_PRO_CATEGORIES
from oe_eval.data.mmlu_tasks import MMLU_SUBJECTS
from oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY
from oe_eval.tasks.oe_eval_tasks.mmlu import create_mmlu_task
from oe_eval.tasks.oe_eval_tasks.mmlu_pro import create_mmlu_pro_task

from tasks import splits_gsm8k
from .tasks import (
    agi_eval,
    hatespeech,
    medmcqa,
    medqa,
    narrativeqa,
    news_gen,
    poem_gen,
    ruler,
    squad,
    squad2,
    sciriff,
    story_gen,
    xsum,
    splits_arc,
    splits_boolq,
    splits_csqa,
    splits_hellaswag,
    splits_openbookqa,
    splits_piqa,
    splits_siqa,
    splits_winogrande,
)


def create_core_mmlu_tasks_withsplits():
    """Creates a dictionary of tasks from a list of subjects.
    Note that the differences between train, validation, and test is declared in TASK_CONFIGS"""
    res = {}
    for sub in MMLU_SUBJECTS:
        res[f"mmlu_{sub}:mc_train"] = create_mmlu_task(sub)
        res[f"mmlu_{sub}:mc_validation"] = create_mmlu_task(sub)
        res[f"mmlu_{sub}:mc_test"] = create_mmlu_task(sub)
        res[f"mmlu_{sub}:rc_train"] = create_mmlu_task(sub, is_mc=False)
        res[f"mmlu_{sub}:rc_validation"] = create_mmlu_task(sub, is_mc=False)
        res[f"mmlu_{sub}:rc_test"] = create_mmlu_task(sub, is_mc=False)
    return res

def create_core_mmlu_pro_tasks_withsplits():
    """Creates a dictionary of MMLU-Pro tasks from a list of categories
    Note that the differences between train, validation, and test is declared in TASK_CONFIGS"""
    res = {}
    for sub in MMLU_PRO_CATEGORIES:
        res[f"mmlu_pro_{sub}:mc_train"] = create_mmlu_pro_task(sub)
        res[f"mmlu_pro_{sub}:mc_validation"] = create_mmlu_pro_task(sub)
        res[f"mmlu_pro_{sub}:mc_test"] = create_mmlu_pro_task(sub)
        res[f"mmlu_pro_{sub}:rc_train"] = create_mmlu_pro_task(sub, is_mc=False)
        res[f"mmlu_pro_{sub}:rc_validation"] = create_mmlu_pro_task(sub, is_mc=False)
        res[f"mmlu_pro_{sub}:rc_test"] = create_mmlu_pro_task(sub, is_mc=False)
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
    "arc_easy:rc_train": splits_arc.ARCEasy_RC_Train,
    "arc_easy:rc_validation": splits_arc.ARCEasy_RC_Validation,
    "arc_easy:rc_test": splits_arc.ARCEasy_RC_Test,
    "arc_challenge:mc_train": splits_arc.ARCChallenge_MC_Train,
    "arc_challenge:mc_validation": splits_arc.ARCChallenge_MC_Validation,
    "arc_challenge:mc_test": splits_arc.ARCChallenge_MC_Test,
    "arc_challenge:rc_train": splits_arc.ARCChallenge_RC_Train,
    "arc_challenge:rc_validation": splits_arc.ARCChallenge_RC_Validation,
    "arc_challenge:rc_test": splits_arc.ARCChallenge_RC_Test,
    "boolq:mc_train": splits_boolq.BoolQ_MC_Train,
    "boolq:mc_validation": splits_boolq.BoolQ_MC_Validation,
    "boolq:mc_test": splits_boolq.BoolQ_MC_Test,
    "boolq:rc_train": splits_boolq.BoolQ_RC_Train,
    "boolq:rc_validation": splits_boolq.BoolQ_RC_Validation,
    "boolq:rc_test": splits_boolq.BoolQ_RC_Test,
    "csqa:mc_train": splits_csqa.CommonsenseQA_MC_Train,
    "csqa:mc_validation": splits_csqa.CommonsenseQA_MC_Validation,
    "csqa:mc_test": splits_csqa.CommonsenseQA_MC_Test,
    "csqa:rc_train": splits_csqa.CommonsenseQA_RC_Train,
    "csqa:rc_validation": splits_csqa.CommonsenseQA_RC_Validation,
    "csqa:rc_test": splits_csqa.CommonsenseQA_RC_Test,
    "hellaswag:mc_train": splits_hellaswag.HellaSwag_MC_Train,
    "hellaswag:mc_validation": splits_hellaswag.HellaSwag_MC_Validation,
    "hellaswag:mc_test": splits_hellaswag.HellaSwag_MC_Test,
    "hellaswag:rc_train": splits_hellaswag.HellaSwag_RC_Train,
    "hellaswag:rc_validation": splits_hellaswag.HellaSwag_RC_Validation,
    "hellaswag:rc_test": splits_hellaswag.HellaSwag_RC_Test,
    "openbookqa:mc_train": splits_openbookqa.OpenBookQA_MC_Train,
    "openbookqa:mc_validation": splits_openbookqa.OpenBookQA_MC_Validation,
    "openbookqa:mc_test": splits_openbookqa.OpenBookQA_MC_Test,
    "openbookqa:rc_train": splits_openbookqa.OpenBookQA_RC_Train,
    "openbookqa:rc_validation": splits_openbookqa.OpenBookQA_RC_Validation,
    "openbookqa:rc_test": splits_openbookqa.OpenBookQA_RC_Test,
    "piqa:mc_train": splits_piqa.PIQA_MC_Train,
    "piqa:mc_validation": splits_piqa.PIQA_MC_Validation,
    "piqa:mc_test": splits_piqa.PIQA_MC_Test,
    "piqa:rc_train": splits_piqa.PIQA_RC_Train,
    "piqa:rc_validation": splits_piqa.PIQA_RC_Validation,
    "piqa:rc_test": splits_piqa.PIQA_RC_Test,
    "socialiqa:mc_train": splits_siqa.SocialIQA_MC_Train,
    "socialiqa:mc_validation": splits_siqa.SocialIQA_MC_Validation,
    "socialiqa:mc_test": splits_siqa.SocialIQA_MC_Test,
    "socialiqa:rc_train": splits_siqa.SocialIQA_RC_Train,
    "socialiqa:rc_validation": splits_siqa.SocialIQA_RC_Validation,
    "socialiqa:rc_test": splits_siqa.SocialIQA_RC_Test,
    "winogrande:mc_train": splits_winogrande.Winogrande_MC_Train,
    "winogrande:mc_validation": splits_winogrande.Winogrande_MC_Validation,
    "winogrande:mc_test": splits_winogrande.Winogrande_MC_Test,
    "winogrande:rc_train": splits_winogrande.Winogrande_RC_Train,
    "winogrande:rc_validation": splits_winogrande.Winogrande_RC_Validation,
    "winogrande:rc_test": splits_winogrande.Winogrande_RC_Test,

    # MMLU
    **create_core_mmlu_tasks_withsplits(),
    **create_core_mmlu_pro_tasks_withsplits(),

    # GSM8K
    "gsm8k:perplexity": splits_gsm8k.GSM8K_Perplexity_Base,

}

TASK_REGISTRY.update(new_task_registry)
