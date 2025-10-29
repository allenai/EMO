from typing import Dict

from oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY

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
    story_gen,
    xsum,
    arc_splits,
)

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
    "arc_easy:mc_train": arc_splits.ARCEasyMC_Train,
    "arc_easy:mc_validation": arc_splits.ARCEasyMC_Validation,
    "arc_easy:mc_test": arc_splits.ARCEasyMC_Test,
    "arc_easy:rc_train": arc_splits.ARCEasyRC_Train,
    "arc_easy:rc_validation": arc_splits.ARCEasyRC_Validation,
    "arc_easy:rc_test": arc_splits.ARCEasyRC_Test,
    "arc_challenge:mc_train": arc_splits.ARCChallengeMC_Train,
    "arc_challenge:mc_validation": arc_splits.ARCChallengeMC_Validation,
    "arc_challenge:mc_test": arc_splits.ARCChallengeMC_Test,
    "arc_challenge:rc_train": arc_splits.ARCChallengeRC_Train,
    "arc_challenge:rc_validation": arc_splits.ARCChallengeRC_Validation,
    "arc_challenge:rc_test": arc_splits.ARCChallengeRC_Test,
}

TASK_REGISTRY.update(new_task_registry)
