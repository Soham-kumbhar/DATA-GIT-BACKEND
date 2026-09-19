import json
from typing import Any

from groq import Groq

from app.core.config import settings


class GroqReportService:

    DEFAULT_MODEL = "openai/gpt-oss-120b"

    # ============================================================
    # EXISTING: ML VERSION COMPARISON ANALYSIS
    # ============================================================

    @staticmethod
    def analyze_comparison(
        comparison: dict[str, Any],
    ) -> dict[str, Any]:

        if not settings.groq_api_key:
            return {
                "status": "unavailable",
                "reason": "Groq API key is not configured.",
                "root_cause": {},
                "recommendations": [],
            }

        try:
            client = Groq(
                api_key=settings.groq_api_key
            )

            evidence = (
                GroqReportService
                ._build_evidence_payload(
                    comparison
                )
            )

            system_prompt = """
You are the AI analysis layer of DataGit, an ML version comparison system.

Your job is to INTERPRET deterministic evidence supplied by DataGit.

The deterministic DataGit comparison is authoritative.
Your job is NOT to calculate facts.
Your job is to explain what the evidence does and does not support.

============================================================
STRICT EVIDENCE RULES
============================================================

1. NEVER invent facts.

2. NEVER calculate, recalculate, estimate, approximate, or derive
   numerical statistics.

3. NEVER change, round, reinterpret, or contradict DataGit numbers.

4. DataGit's deterministic evidence is authoritative for observed facts.

5. Clearly distinguish:
   - observed facts
   - possible contributors
   - hypotheses
   - limitations
   - recommendations

6. Every contributor must cite the supplied evidence supporting it.

7. Do NOT claim causality unless the supplied evidence explicitly
   establishes a causal relationship.

8. A dataset change occurring in the same comparison as a performance
   change does NOT prove that the dataset caused the performance change.

9. A performance change occurring after a dataset change is temporal
   coincidence unless the evidence explicitly connects the changed
   dataset to model training and the changed evaluation.

10. If retraining status is unknown, NEVER assume the changed dataset
    was used to retrain the model.

11. If retraining status is unknown, a dataset change MUST NOT be
    classified as a negative or positive causal contributor merely
    because performance changed.

12. If the evidence only shows:
    - dataset changed
    - performance changed
    - code did not change
    - parameters did not change
    - retraining status is unknown

    then the dataset contributor direction MUST be:

    "unknown"

    and the root-cause conclusion MUST state that the exact cause
    cannot be determined from the available evidence.

13. Do NOT force a root cause when the evidence does not support one.

14. When evidence is insufficient, explicitly use language such as:

    "The root cause cannot be determined from the available evidence."

15. Confidence values may ONLY be:
    - high
    - medium
    - low

============================================================
MODEL NAME / CODE RULES
============================================================

16. A changed model_name string alone does NOT prove:
    - model implementation changed
    - architecture changed
    - weights changed
    - training code changed
    - preprocessing changed
    - model behavior changed

17. Never use a model name change as proof of a different model
    implementation.

18. Never use a model name change as an alternative explanation for
    a performance change unless explicit evidence connects the name
    change to a real implementation or configuration change.

19. If DataGit reports code_changed = false, say:

    "No tracked model/training-code changes were detected."

    Do NOT say that hidden implementation changes definitely did
    or did not occur.

20. Git commit changes do NOT prove model or training-code changes.

21. Use changed_files and code_changed_files when deciding whether
    model/training code changed.

22. If no model/training code files changed, do not claim a code
    change caused the performance change.

============================================================
HIDDEN INFORMATION RULES
============================================================

23. NEVER infer hidden:
    - preprocessing changes
    - runtime changes
    - library changes
    - environment changes
    - configuration changes
    - model weights
    - architecture changes
    - artifacts
    - random seeds
    - training behavior

    unless these are explicitly supplied as evidence.

24. Do NOT list hidden preprocessing, runtime, library, environment,
    configuration, weights, architecture, or artifacts as concrete
    alternative explanations.

25. Instead, when appropriate, say:

    "The available evidence does not include information about
    preprocessing, runtime environment, model artifacts, or other
    hidden implementation details."

26. Missing evidence is NOT evidence of a hidden change.

27. Missing evidence must never be converted into a factual claim.

============================================================
RETRAINING RULES
============================================================

28. NEVER claim retraining occurred unless DataGit explicitly records
    retraining.

29. NEVER claim retraining did not occur unless DataGit explicitly
    records that it did not occur.

30. If retraining status is missing, explicitly state:

    "Retraining status is not recorded."

31. Do NOT infer that changed training data was actually consumed by
    a model.

32. Do NOT infer that a changed dataset was excluded from evaluation.

33. Do NOT infer that newly added rows were evaluated or not evaluated
    unless the evidence explicitly states this.

============================================================
EVALUATION RULES
============================================================

34. Never infer the semantic role of an evaluation dataset.

35. prediction_count does NOT prove whether predictions came from:
    - training data
    - validation data
    - test data
    - holdout data
    - production data

36. Never claim that the evaluation dataset was fixed unless DataGit
    explicitly provides evidence identifying it as fixed.

37. Never infer missing confusion-matrix values.

38. Never infer precision, recall, F1, error rate, or other metrics
    that are missing from either version.

39. If evaluation evidence exists for only one version, explicitly say
    that a before/after error comparison cannot be determined.

40. If evaluation evidence changed, state ONLY that the recorded
    evaluation evidence changed.

41. Do NOT assume why the evaluation changed.

42. If misclassified samples are supplied, use only those supplied
    samples.

============================================================
PERFORMANCE RULES
============================================================

43. If performance metrics changed, report the observed change exactly.

44. Never invent a reason for a performance change.

45. If both performance and evaluation changed, do NOT automatically
    attribute the change to the dataset.

46. If accuracy decreases, state the observed accuracy decrease.

47. If false positives or false negatives change, state those observed
    changes exactly as provided by DataGit.

48. Do NOT calculate new metrics from the confusion matrix.

49. Do NOT independently verify or recalculate DataGit metrics.

============================================================
DATASET RULES
============================================================

50. A dataset change may be described as an observed change.

51. A dataset change is NOT automatically a causal contributor.

52. If dataset changed and performance changed but retraining status
    is unknown, use:

    direction = "unknown"

    unless explicit evidence connects the dataset to training and the
    observed performance result.

53. If dataset changed but performance did not change, describe the
    dataset as a possible or neutral factor, NOT as proven causal
    evidence.

54. Do not claim dataset quality caused a result unless the deterministic
    evidence explicitly supports the claim.

============================================================
ROOT-CAUSE DECISION RULES
============================================================

55. Root cause means an evidence-supported explanation, not simply
    something that changed.

56. A factor changing at the same time as performance does NOT make it
    a root cause.

57. When evidence is insufficient, contributors may be:

    direction = "unknown"

58. Do not assign direction "negative" to the dataset merely because
    performance decreased after the dataset changed.

59. Do not assign direction "positive" to the dataset merely because
    performance improved after the dataset changed.

60. Do not use "high" confidence for causal claims when retraining,
    model artifacts, or implementation changes are unknown.

61. When no causal contributor can be established, overall confidence
    should normally be "low" or "medium", depending on the strength
    of the observed facts.

62. An observed regression is NOT the same thing as a known root cause.

63. Use this distinction:

    Observed fact:
    "Accuracy decreased from the recorded before value to the
    recorded after value."

    Possible contributor:
    "The dataset changed between the versions."

    Causal conclusion:
    "The root cause cannot be determined from the available evidence."

============================================================
ALTERNATIVE EXPLANATION RULES
============================================================

64. Alternative explanations must be grounded in supplied evidence.

65. Do NOT invent hidden causes.

66. Do NOT present any of the following as an actual alternative
    explanation unless DataGit explicitly provides evidence for it:
    - hidden preprocessing changes
    - runtime changes
    - library changes
    - environment changes
    - hidden configuration changes
    - model weight changes
    - architecture changes
    - hidden artifacts

67. When information about preprocessing, runtime, libraries,
    environment, configuration, model weights, architecture, or
    artifacts is absent, put that information under "limitations",
    NOT under "alternative_explanations".

68. Missing evidence is a limitation, not an alternative explanation.

69. Example of correct wording:

    limitation:
    "The comparison does not contain model artifact or runtime
    environment evidence."

    Do NOT write:

    "The runtime environment may have caused the regression."

70. Alternative explanations should only describe evidence-supported
    possibilities that are actually present in the DataGit evidence.

============================================================
RECOMMENDATION RULES
============================================================

68. Recommendations must be tied to actual evidence.

69. Recommendations may identify missing evidence, but clearly label
    it as missing evidence.

70. Do NOT recommend fixing a cause that has not been established.

71. Good recommendations may include:
    - record retraining events
    - record model artifact identity
    - version evaluation data
    - record training configuration
    - link model artifacts to DataGit versions
    - capture evaluation dataset identity

72. Recommendations must not imply that the recommended missing
    evidence is currently known to exist.

============================================================
SPECIAL RULE FOR THE CURRENT TYPE OF TEST
============================================================

73. When the supplied evidence shows:

    dataset changed
    AND
    performance decreased
    AND
    evaluation changed
    AND
    code did not change
    AND
    parameters did not change
    AND
    retraining status is unknown

    then DO NOT say:

    "The dataset caused the regression."

    Do NOT say:

    "The added training rows caused the regression."

    Do NOT set the dataset contributor direction to "negative"
    solely from those facts.

    Instead say that:

    - the dataset changed
    - performance changed
    - evaluation changed
    - no tracked model/training-code change was detected
    - parameter changes were not recorded
    - retraining status is not recorded
    - therefore the exact root cause cannot be determined from
      the available evidence

74. In that situation, a dataset contributor should normally be:

    {
      "direction": "unknown",
      "confidence": "low"
    }

    unless explicit evidence provides a stronger causal connection.

============================================================
OUTPUT REQUIREMENTS
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{
  "root_cause": {
    "summary": "",
    "contributors": [
      {
        "factor": "",
        "direction": "positive|negative|neutral|unknown",
        "evidence": [],
        "reasoning": "",
        "confidence": "high|medium|low"
      }
    ],
    "overall_confidence": "high|medium|low",
    "limitations": [],
    "alternative_explanations": []
  },
  "recommendations": [
    {
      "recommendation": "",
      "reason": "",
      "priority": "high|medium|low"
    }
  ]
}
"""

            user_prompt = (
                "Analyze the following deterministic DataGit evidence.\n\n"
                "Do not calculate new statistics.\n"
                "Do not infer missing information.\n"
                "Use only the supplied evidence.\n\n"
                "The deterministic DataGit layer is authoritative.\n"
                "Missing information is not evidence.\n"
                "Do not fill gaps with assumptions.\n"
                "Do not force a root cause when the evidence is insufficient.\n\n"
                + json.dumps(
                    evidence,
                    indent=2,
                    default=str,
                )
            )

            response = client.chat.completions.create(
                model=GroqReportService.DEFAULT_MODEL,
                temperature=0.2,
                max_completion_tokens=1000,
                response_format={
                    "type": "json_object"
                },
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            content = (
                response.choices[0]
                .message
                .content
            )

            result = json.loads(
                content
            )

            return {
                "status": "success",
                "root_cause":
                    result.get(
                        "root_cause",
                        {},
                    ),
                "recommendations":
                    result.get(
                        "recommendations",
                        [],
                    ),
            }

        except Exception as exc:

            return {
                "status": "error",
                "reason": str(exc),
                "root_cause": {},
                "recommendations": [],
            }

    # ============================================================
    # DATASET PREPARATION AI ANALYSIS
    # ============================================================

    @staticmethod
    def analyze_dataset_preparation(
        report: dict[str, Any],
    ) -> dict[str, Any]:

        if not settings.groq_api_key:
            return {
                "status": "unavailable",
                "reason": "Groq API key is not configured.",
                "summary": "",
                "quality_assessment": "",
                "changes_explained": [],
                "recommendations": [],
            }

        try:
            client = Groq(
                api_key=settings.groq_api_key
            )

            system_prompt = """
You are the AI analysis layer of DataGit Dataset Preparation.

Your job is to INTERPRET the deterministic Dataset Preparation
Report supplied by DataGit.

The deterministic DataGit report is authoritative.

Your job is NOT to calculate facts.
Your job is to explain the supplied facts clearly.

============================================================
STRICT EVIDENCE RULES
============================================================

1. NEVER invent facts.

2. NEVER calculate, recalculate, estimate, approximate, or derive
   numerical statistics.

3. NEVER change, round, reinterpret, or contradict DataGit numbers.

4. Treat DataGit's deterministic report as authoritative.

5. Only discuss information explicitly supplied in the report.

6. Clearly distinguish observed facts from recommendations.

7. Do not claim that an operation caused a change unless
   the supplied evidence explicitly establishes causality.

8. When a before/after metric changes, describe only the
   observed change unless the report explicitly identifies
   the operation responsible for that change.

9. For example, if outlier count changes from 0 to 1 after
   preparation, say:

   "The outlier count increased from 0 to 1."

   Do NOT say:

   "The imputation introduced an outlier."

10. Do not attribute a before/after change to a specific
    preparation operation merely because that operation occurred
    before the observed change.

11. Do not invent business meaning for columns.

12. Do not invent reasons for missing values, duplicates, or outliers.

13. Do not claim that the dataset is "ready for production" unless
    the supplied evidence explicitly supports that statement.

14. Recommendations must be connected to supplied evidence.

15. Missing information is a limitation, not evidence of a problem.

16. Never expose or reproduce API keys, secrets, or credentials.

============================================================
DATA PREPARATION INTERPRETATION
============================================================

You may explain:

- rows added or removed
- columns added or removed
- missing values before and after
- duplicate rows before and after
- outlier counts before and after
- constant columns
- column type changes
- preparation operations
- operation parameters
- output format
- reproducibility information

You may describe whether the supplied evidence shows improvement,
no change, or a change that needs attention.

Do not perform new calculations.

============================================================
RECOMMENDATION SAFETY
============================================================

Recommendations must describe only observed conditions.

When recommending investigation of an outlier, use wording such as:

"Investigate the outlier detected after preparation."

Do NOT use wording such as:

"Investigate the newly introduced outlier."

Do not claim that a preparation operation created, introduced,
caused, or produced an observed quality issue unless the supplied
report explicitly establishes that relationship.

============================================================
OUTPUT REQUIREMENTS
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{
  "summary": "",
  "quality_assessment": "",
  "changes_explained": [
    {
      "change": "",
      "explanation": ""
    }
  ],
  "recommendations": [
    {
      "recommendation": "",
      "reason": "",
      "priority": "high|medium|low"
    }
  ]
}
"""

            user_prompt = (
                "Analyze the following deterministic DataGit "
                "Dataset Preparation Report.\n\n"
                "The DataGit report is authoritative.\n"
                "Do not calculate new statistics.\n"
                "Do not infer missing information.\n"
                "Do not invent causes.\n\n"
                + json.dumps(
                    report,
                    indent=2,
                    default=str,
                )
            )

            response = client.chat.completions.create(
                model=GroqReportService.DEFAULT_MODEL,
                temperature=0.2,
                max_completion_tokens=1000,
                response_format={
                    "type": "json_object"
                },
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            content = (
                response.choices[0]
                .message
                .content
            )

            result = json.loads(content)

            return {
                "status": "success",
                "summary": result.get(
                    "summary",
                    "",
                ),
                "quality_assessment": result.get(
                    "quality_assessment",
                    "",
                ),
                "changes_explained": result.get(
                    "changes_explained",
                    [],
                ),
                "recommendations": result.get(
                    "recommendations",
                    [],
                ),
            }

        except Exception as exc:
            return {
                "status": "error",
                "reason": str(exc),
                "summary": "",
                "quality_assessment": "",
                "changes_explained": [],
                "recommendations": [],
            }

    # ============================================================
    # BUILD DETERMINISTIC EVIDENCE FOR GROQ
    # ============================================================

    @staticmethod
    def _build_evidence_payload(
        comparison: dict[str, Any],
    ) -> dict[str, Any]:

        ml_comparison = (
            comparison.get(
                "ml_comparison"
            )
            or {}
        )

        evaluation = (
            ml_comparison.get(
                "evaluation"
            )
            or {}
        )

        dataset_analysis = (
            comparison.get(
                "dataset_analysis"
            )
            or {}
        )

        dataset_diff = (
            comparison.get(
                "dataset_diff"
            )
            or {}
        )

        performance = (
            comparison.get(
                "performance"
            )
            or {}
        )

        return {
            # ----------------------------------------------------
            # VERSION IDENTITY
            # ----------------------------------------------------

            "versions": {
                "version_1":
                    comparison.get(
                        "version_1"
                    ),

                "version_2":
                    comparison.get(
                        "version_2"
                    ),
            },

            # ----------------------------------------------------
            # GIT
            # ----------------------------------------------------

            "git": {
                "changed":
                    comparison.get(
                        "git_changed",
                        False,
                    ),

                "commit_before":
                    comparison.get(
                        "git_commit_before"
                    ),

                "commit_after":
                    comparison.get(
                        "git_commit_after"
                    ),

                "changed_files":
                    comparison.get(
                        "changed_files",
                        [],
                    ),

                "code_changed":
                    comparison.get(
                        "code_changed",
                        False,
                    ),

                "code_changed_files":
                    comparison.get(
                        "code_changed_files",
                        [],
                    ),

                "code_patch":
                    comparison.get(
                        "code_patch",
                        "",
                    ),
            },

            # ----------------------------------------------------
            # DVC / DATASET
            # ----------------------------------------------------

            "dataset": {
                "dvc_changed":
                    comparison.get(
                        "dvc_changed",
                        False,
                    ),

                "diff":
                    dataset_diff,

                "analysis":
                    dataset_analysis,
            },

            # ----------------------------------------------------
            # MODEL
            # ----------------------------------------------------

            "model": {
                "model_name_before":
                    ml_comparison.get(
                        "model_name_before"
                    ),

                "model_name_after":
                    ml_comparison.get(
                        "model_name_after"
                    ),

                "features_before":
                    ml_comparison.get(
                        "features_before",
                        [],
                    ),

                "features_after":
                    ml_comparison.get(
                        "features_after",
                        [],
                    ),

                "features_added":
                    ml_comparison.get(
                        "features_added",
                        [],
                    ),

                "features_removed":
                    ml_comparison.get(
                        "features_removed",
                        [],
                    ),

                "parameters_before":
                    ml_comparison.get(
                        "parameters_before",
                        {},
                    ),

                "parameters_after":
                    ml_comparison.get(
                        "parameters_after",
                        {},
                    ),

                "parameter_changes":
                    ml_comparison.get(
                        "parameter_changes",
                        {},
                    ),
            },

            # ----------------------------------------------------
            # PERFORMANCE
            # ----------------------------------------------------

            "performance": {
                "before":
                    performance.get(
                        "metrics_before",
                        {},
                    ),

                "after":
                    performance.get(
                        "metrics_after",
                        {},
                    ),

                "changes":
                    performance.get(
                        "metric_changes",
                        {},
                    ),

                "changed":
                    performance.get(
                        "performance_changed",
                        False,
                    ),
            },

            # ----------------------------------------------------
            # EVALUATION / ERROR EVIDENCE
            # ----------------------------------------------------

            "evaluation": {
                "available":
                    evaluation.get(
                        "available",
                        False,
                    ),

                "status":
                    evaluation.get(
                        "status"
                    ),

                "changed":
                    evaluation.get(
                        "changed",
                        False,
                    ),

                "summary":
                    evaluation.get(
                        "summary"
                    ),

                "before":
                    evaluation.get(
                        "before"
                    ),

                "after":
                    evaluation.get(
                        "after"
                    ),

                "changes":
                    evaluation.get(
                        "changes",
                        [],
                    ),

                "confusion_matrix":
                    evaluation.get(
                        "confusion_matrix"
                    ),

                "misclassified_samples":
                    evaluation.get(
                        "misclassified_samples"
                    ),
            },

            # ----------------------------------------------------
            # EVIDENCE CHAIN
            # ----------------------------------------------------

            "evidence_chain":
                comparison.get(
                    "evidence_chain",
                    [],
                ),
        }