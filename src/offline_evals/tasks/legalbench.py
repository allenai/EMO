"""
LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in LLMs
https://huggingface.co/datasets/nguha/legalbench

Implements ~110 classification tasks using RC (Ranked Classification) evaluation.
"""

from oe_eval.tasks.base_task import MultipleChoiceTask
from oe_eval.tasks.utils import make_cloze_prompt
from oe_eval.utils import get_dict_with_defaults


class GenericLegalBenchRC(MultipleChoiceTask):
    """
    Generic base class for LegalBench classification tasks using RC evaluation.

    RC (Ranked Classification): Scores each answer choice via log-likelihood
    of the continuation, normalized by character count.

    Subclasses should set:
    - TASK_NAME: str - unique task identifier
    - CHOICES: list[str] - answer choices for classification
    - TASK_CONFIG_DEFAULTS: dict - task-specific config (esp. dataset_name)
    - TEXT_FIELD: str - field name containing input text (default: "text")
    - ANSWER_FIELD: str - field name containing answer label (default: "answer")
    """

    VERSION = 0
    TASK_NAME = None  # Set by subclass
    CHOICES = None  # Set by subclass
    TEXT_FIELD = "text"  # Field containing input text
    ANSWER_FIELD = "answer"  # Field containing answer label

    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "nguha/legalbench",
        "dataset_name": None,  # Must be set by subclass
        "native_id_field": "index",
        "primary_metric": "acc_per_char",  # Length-normalized log-likelihood scoring
        "split": "test",
        "fewshot_source": "train",
    }

    def has_training_docs(self):
        # Most LegalBench tasks have train split, but check dynamically if needed
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            # Check if train split exists
            if "train" in self.dataset:
                self._training_docs = list(map(self._process_doc, self.dataset["train"]))
            else:
                self._training_docs = []
        return self._training_docs

    def validation_docs(self):
        return NotImplemented

    def test_docs(self):
        # Use the split specified in task_config, default to "test"
        split = self.task_config.get("split", "test")
        return map(self._process_doc, self.dataset[split])

    def _process_doc(self, doc):
        """
        Process a document into evaluation format.

        Uses RC (Ranked Classification) style: creates a cloze prompt and scores
        each answer choice as a direct continuation.
        """
        # Get text content using class attribute
        text = doc[self.TEXT_FIELD]

        # Create cloze prompt with no prefix (direct continuation)
        query = make_cloze_prompt(text, question_prefix="")

        out_doc = {
            "index": doc.get("index", 0),
            "query": query,
            "choices": self.CHOICES,
            "gold": self.CHOICES.index(doc[self.ANSWER_FIELD]),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]

    def unconditioned_prompt(self):
        return "Answer:"


class LegalBenchAbercrombie(GenericLegalBenchRC):
    """
    Backward compatibility: Abercrombie task (trademark distinctiveness classification).
    Categories: generic, descriptive, suggestive, arbitrary, fanciful
    """

    TASK_NAME = "legalbench_abercrombie"
    CHOICES = ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"]

    TASK_CONFIG_DEFAULTS: dict = get_dict_with_defaults(
        {"dataset_name": "abercrombie"},
        GenericLegalBenchRC.TASK_CONFIG_DEFAULTS,
    )
# Dictionary of all LegalBench classification tasks
# Generated from nguha/legalbench dataset inspection
LEGALBENCH_CLASSIFICATION_TASKS = {
    "abercrombie": {
        "choices": ['arbitrary', 'descriptive', 'fanciful', 'generic', 'suggestive'],
    },
    "proa": {
        "choices": ['No', 'Yes'],
    },
    "definition_classification": {
        "choices": ['No', 'Yes'],
    },
    "contract_qa": {
        "choices": ['No', 'Yes'],
    },
    "hearsay": {
        "choices": ['No', 'Yes'],
    },
    "learned_hands_benefits": {
        "choices": ['No', 'Yes'],
    },
    "learned_hands_business": {
        "choices": ['No', 'Yes'],
    },
    "learned_hands_consumer": {
        "choices": ['No', 'Yes'],
    },
    "learned_hands_courts": {
        "choices": ['No', 'Yes'],
    },
    "learned_hands_crime": {
        "choices": ['No', 'Yes'],
    },
    "learned_hands_divorce": {
        "choices": ['No', 'Yes'],
    },
    "learned_hands_domestic_violence": {
        "choices": ['No', 'Yes'],
    },
    "learned_hands_education": {
        "choices": ['No', 'Yes'],
    },
    "learned_hands_employment": {
        "choices": ['No', 'Yes'],
    },
    "learned_hands_estates": {
        "choices": ['No', 'Yes'],
    },
    "learned_hands_family": {
        "choices": ['No', 'Yes'],
    },
    "learned_hands_health": {
        "choices": ['No', 'Yes'],
    },
    "learned_hands_housing": {
        "choices": ['No', 'Yes'],
    },
    "learned_hands_immigration": {
        "choices": ['No', 'Yes'],
    },
    "learned_hands_torts": {
        "choices": ['No', 'Yes'],
    },
    "legal_reasoning_causality": {
        "choices": ['No', 'Yes'],
    },
    "maud_ability_to_consummate_concept_is_subject_to_mae_carveouts": {
        "choices": ['A', 'B'],
    },
    "maud_financial_point_of_view_is_the_sole_consideration": {
        "choices": ['A', 'B'],
    },
    "maud_accuracy_of_fundamental_target_rws_bringdown_standard": {
        "choices": ['A', 'B', 'C'],
    },
    "maud_accuracy_of_target_general_rw_bringdown_timing_answer": {
        "choices": ['A', 'B'],
    },
    "maud_change_in_law__subject_to_disproportionate_impact_modifier": {
        "choices": ['A', 'B'],
    },
    "maud_changes_in_gaap_or_other_accounting_principles__subject_to_disproportionate_impact_modifier": {
        "choices": ['A', 'B'],
    },
    "maud_cor_permitted_in_response_to_intervening_event": {
        "choices": ['A', 'B'],
    },
    "maud_cor_permitted_with_board_fiduciary_determination_only": {
        "choices": ['A', 'B'],
    },
    "maud_general_economic_and_financial_conditions_subject_to_disproportionate_impact_modifier": {
        "choices": ['A', 'B'],
    },
    "maud_includes_consistent_with_past_practice": {
        "choices": ['A', 'B'],
    },
    "maud_intervening_event_-_required_to_occur_after_signing_-_answer": {
        "choices": ['A', 'B'],
    },
    "maud_knowledge_definition": {
        "choices": ['A', 'B'],
    },
    "maud_ordinary_course_efforts_standard": {
        "choices": ['A', 'B', 'C'],
    },
    "maud_specific_performance": {
        "choices": ['A', 'B'],
    },
    "maud_tail_period_length": {
        "choices": ['A', 'C', 'D', 'E'],
    },
    "maud_type_of_consideration": {
        "choices": ['A', 'B', 'C', 'D'],
    },
    "overruling": {
        "choices": ['No', 'Yes'],
    },
    "personal_jurisdiction": {
        "choices": ['No', 'Yes'],
    },
    "privacy_policy_entailment": {
        "choices": ['Correct', 'Incorrect'],
    },
    "privacy_policy_qa": {
        "choices": ['Irrelevant', 'Relevant'],
    },
    "supply_chain_disclosure_best_practice_accountability": {
        "choices": ['No', 'Yes'],
    },
    "supply_chain_disclosure_best_practice_training": {
        "choices": ['No', 'Yes'],
    },
    "supply_chain_disclosure_disclosed_accountability": {
        "choices": ['No', 'Yes'],
    },
    "supply_chain_disclosure_disclosed_audits": {
        "choices": ['No', 'Yes'],
    },
    "supply_chain_disclosure_disclosed_certification": {
        "choices": ['No', 'Yes'],
    },
    "supply_chain_disclosure_disclosed_training": {
        "choices": ['No', 'Yes'],
    },
    "supply_chain_disclosure_disclosed_verification": {
        "choices": ['No', 'Yes'],
    },
    "telemarketing_sales_rule": {
        "choices": ['No', 'Yes'],
    },
    "textualism_tool_plain": {
        "choices": ['No', 'Yes'],
    },
    "unfair_tos": {
        "choices": ['Arbitration', 'Choice of law', 'Content removal', 'Contract by using', 'Jurisdiction', 'Limitation of liability', 'Other', 'Unilateral change', 'Unilateral termination'],
    },
    "cuad_affiliate_license-licensee": {
        "choices": ['No', 'Yes'],
    },
    "cuad_affiliate_license-licensor": {
        "choices": ['No', 'Yes'],
    },
    "cuad_anti-assignment": {
        "choices": ['No', 'Yes'],
    },
    "cuad_audit_rights": {
        "choices": ['No', 'Yes'],
    },
    "cuad_cap_on_liability": {
        "choices": ['No', 'Yes'],
    },
    "cuad_change_of_control": {
        "choices": ['No', 'Yes'],
    },
    "cuad_competitive_restriction_exception": {
        "choices": ['No', 'Yes'],
    },
    "cuad_covenant_not_to_sue": {
        "choices": ['No', 'Yes'],
    },
    "cuad_effective_date": {
        "choices": ['No', 'Yes'],
    },
    "cuad_exclusivity": {
        "choices": ['No', 'Yes'],
    },
    "cuad_expiration_date": {
        "choices": ['No', 'Yes'],
    },
    "cuad_governing_law": {
        "choices": ['No', 'Yes'],
    },
    "cuad_insurance": {
        "choices": ['No', 'Yes'],
    },
    "cuad_ip_ownership_assignment": {
        "choices": ['No', 'Yes'],
    },
    "cuad_irrevocable_or_perpetual_license": {
        "choices": ['No', 'Yes'],
    },
    "cuad_joint_ip_ownership": {
        "choices": ['No', 'Yes'],
    },
    "cuad_license_grant": {
        "choices": ['No', 'Yes'],
    },
    "cuad_liquidated_damages": {
        "choices": ['No', 'Yes'],
    },
    "cuad_minimum_commitment": {
        "choices": ['No', 'Yes'],
    },
    "cuad_most_favored_nation": {
        "choices": ['No', 'Yes'],
    },
    "cuad_no-solicit_of_customers": {
        "choices": ['No', 'Yes'],
    },
    "cuad_no-solicit_of_employees": {
        "choices": ['No', 'Yes'],
    },
    "cuad_non-compete": {
        "choices": ['No', 'Yes'],
    },
    "cuad_non-disparagement": {
        "choices": ['No', 'Yes'],
    },
    "cuad_non-transferable_license": {
        "choices": ['No', 'Yes'],
    },
    "cuad_notice_period_to_terminate_renewal": {
        "choices": ['No', 'Yes'],
    },
    "cuad_post-termination_services": {
        "choices": ['No', 'Yes'],
    },
    "cuad_price_restrictions": {
        "choices": ['No', 'Yes'],
    },
    "cuad_renewal_term": {
        "choices": ['No', 'Yes'],
    },
    "cuad_revenue-profit_sharing": {
        "choices": ['No', 'Yes'],
    },
    "cuad_rofr-rofo-rofn": {
        "choices": ['No', 'Yes'],
    },
    "cuad_source_code_escrow": {
        "choices": ['No', 'Yes'],
    },
    "cuad_termination_for_convenience": {
        "choices": ['No', 'Yes'],
    },
    "cuad_third_party_beneficiary": {
        "choices": ['No', 'Yes'],
    },
    "cuad_uncapped_liability": {
        "choices": ['No', 'Yes'],
    },
    "cuad_unlimited-all-you-can-eat-license": {
        "choices": ['No', 'Yes'],
    },
    "cuad_volume_restriction": {
        "choices": ['No', 'Yes'],
    },
    "cuad_warranty_duration": {
        "choices": ['No', 'Yes'],
    },
    "contract_nli_confidentiality_of_agreement": {
        "choices": ['No', 'Yes'],
    },
    "contract_nli_explicit_identification": {
        "choices": ['No', 'Yes'],
    },
    "contract_nli_inclusion_of_verbally_conveyed_information": {
        "choices": ['No', 'Yes'],
    },
    "contract_nli_limited_use": {
        "choices": ['No', 'Yes'],
    },
    "contract_nli_no_licensing": {
        "choices": ['No', 'Yes'],
    },
    "contract_nli_notice_on_compelled_disclosure": {
        "choices": ['No', 'Yes'],
    },
    "contract_nli_permissible_acquirement_of_similar_information": {
        "choices": ['No', 'Yes'],
    },
    "contract_nli_permissible_copy": {
        "choices": ['No', 'Yes'],
    },
    "contract_nli_permissible_development_of_similar_information": {
        "choices": ['No', 'Yes'],
    },
    "contract_nli_permissible_post-agreement_possession": {
        "choices": ['No', 'Yes'],
    },
    "contract_nli_return_of_confidential_information": {
        "choices": ['No', 'Yes'],
    },
    "contract_nli_sharing_with_employees": {
        "choices": ['No', 'Yes'],
    },
    "contract_nli_sharing_with_third-parties": {
        "choices": ['No', 'Yes'],
    },
    "contract_nli_survival_of_obligations": {
        "choices": ['No', 'Yes'],
    },
    "citation_prediction_classification": {
        "choices": ['No', 'Yes'],
    },
    "diversity_1": {
        "choices": ['No', 'Yes'],
    },
    "diversity_2": {
        "choices": ['No', 'Yes'],
    },
    "diversity_3": {
        "choices": ['No', 'Yes'],
    },
    "diversity_4": {
        "choices": ['No', 'Yes'],
    },
    "diversity_5": {
        "choices": ['No', 'Yes'],
    },
    "diversity_6": {
        "choices": ['No', 'Yes'],
    },
}


def create_legalbench_task(task_name: str, task_metadata: dict):
    """Factory function to create a LegalBench task class dynamically."""
    choices = task_metadata["choices"]
    text_field = task_metadata.get("text_field", "text")
    answer_field = task_metadata.get("answer_field", "answer")

    class LegalBenchRC(GenericLegalBenchRC):
        TASK_NAME = f"legalbench_{task_name}"
        CHOICES = choices
        TEXT_FIELD = text_field
        ANSWER_FIELD = answer_field
        TASK_CONFIG_DEFAULTS = get_dict_with_defaults(
            {"dataset_name": task_name},
            GenericLegalBenchRC.TASK_CONFIG_DEFAULTS,
        )

    return LegalBenchRC


def create_legalbench_tasks() -> dict:
    """
    Create all LegalBench classification tasks.

    Returns a dict mapping task keys (e.g., 'legalbench_hearsay:rc') to task classes.
    This supports registering all ~110 classification tasks at once.
    """
    all_tasks = {}

    # Register individual tasks
    for task_name, metadata in LEGALBENCH_CLASSIFICATION_TASKS.items():
        task_key = f"legalbench_{task_name}:rc"
        all_tasks[task_key] = create_legalbench_task(task_name, metadata)

    return all_tasks
