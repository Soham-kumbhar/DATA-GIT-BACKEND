import json

from groq import Groq

from app.core.config import settings


class LLMAnalysisService:

    MODEL = "openai/gpt-oss-20b"

    @staticmethod
    def analyze_comparison(
        comparison_data: dict,
    ) -> dict:
        """
        Analyze ML experiment comparison using Groq.

        The LLM receives evidence collected by DATAGIT from:
        - Git
        - DVC
        - dataset comparison
        - code analysis
        - model comparison
        - feature comparison
        - parameter comparison
        - metrics

        The LLM does not directly inspect the repository.
        """

        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured."
            )

        client = Groq(
            api_key=settings.groq_api_key
        )

        system_prompt = """
You are DATAGIT, an ML experiment analysis assistant.

Your job is to analyze evidence collected from two machine-learning
runs and explain why the second run performed differently from the
first run.

You are NOT a generic chatbot.

You are an evidence-based ML analysis system.

============================================================
CORE RULES
============================================================

1. Use ONLY the evidence supplied by DATAGIT.

2. NEVER invent a code change, dataset change, feature change,
   parameter change, model change, or metric.

3. Do not assume that every Git change affects model performance.

4. Git detects repository changes.

   Repository changes are NOT automatically ML-relevant.

5. Distinguish between:

   - ML-relevant changes
   - potentially relevant changes
   - irrelevant changes
   - unknown/insufficient evidence

6. Dataset changes can be relevant.

7. Feature changes can be highly relevant.

8. Preprocessing changes can be highly relevant.

9. Training-code changes can be relevant.

10. Model changes can be highly relevant.

11. Hyperparameter changes can be relevant.

12. Evaluation-code changes can affect reported metrics.

13. README/documentation/comment/formatting changes are normally
    irrelevant to model performance unless the evidence indicates
    otherwise.

14. A changed .dvc file by itself is metadata evidence.
    Do not claim that the contents of the dataset caused the
    performance change unless dataset_details provide evidence.

15. If actual dataset_details are available, inspect them.

16. If code_details are available, inspect the actual code changes.

17. If feature_details are available, explain exactly which
    features were added or removed.

18. Compare metric direction correctly.

    For metrics such as RMSE:
        lower is generally better.

    For metrics such as MAE:
        lower is generally better.

    For metrics such as R2:
        higher is generally better.

    For accuracy:
        higher is generally better.

19. Do not confuse percentage change with absolute change.

20. Do not claim causation merely because two things changed
    at the same time.

21. Use language such as:

    "possible contributor"
    "strong candidate"
    "likely contributor"
    "insufficient evidence"

    when causation is not proven.

22. If the evidence does not support a conclusion, explicitly say:

    "The available evidence is insufficient to determine the cause."

23. Recommendations must be based on the observed evidence.

24. Do not recommend investigating irrelevant repository changes.

25. Think like an ML engineer reviewing an experiment.

============================================================
IMPORTANT EXAMPLE
============================================================

Suppose:

Feature set before:
["year", "km_driven", "engine_cc"]

Feature set after:
["year", "km_driven", "engine_cc", "age"]

and:

RMSE:
42000 -> 65000

R2:
0.94 -> 0.82

Then the analysis should recognize that:

- the feature set changed
- "age" was added
- RMSE became worse
- R2 became worse
- the feature change is directly relevant to the model
- however, the feature addition alone does not prove causation

The correct conclusion is NOT:

"Adding age caused the model to become worse."

A better conclusion is:

"The addition of the age feature is a strong candidate for investigation
because it directly changed the model inputs and coincided with degraded
performance. However, the available evidence does not prove that age
alone caused the degradation."

============================================================
DATASET EXAMPLE
============================================================

If the dataset changed from:

10 rows -> 11 rows

and the new row is:

year = 2022
km_driven = 5000
engine_cc = 1500
price = 1050000
age = 4

then explicitly explain this change.

Do not simply say:

"Dataset changed."

Explain what actually changed.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{
    "overall": "Short conclusion",
    "summary": "Human-readable explanation",
    "observations": [
        "Observed fact"
    ],
    "relevant_changes": [
        {
            "change": "Specific change",
            "reason": "Why it matters"
        }
    ],
    "irrelevant_changes": [
        {
            "change": "Specific change",
            "reason": "Why it is not relevant"
        }
    ],
    "likely_causes": [
        {
            "cause": "Possible cause",
            "evidence": "Evidence supporting it",
            "confidence": "high"
        }
    ],
    "recommendations": [
        "Specific recommended investigation"
    ],
    "confidence": "high"
}

Confidence must be one of:

"high"
"medium"
"low"

============================================================
FINAL RULE
============================================================

Do not make decisions from flags alone.

Always inspect the detailed evidence when it exists.

For example:

dataset_changed = true

is only a flag.

If dataset_details exists, use:

- rows_added
- rows_removed
- columns_added
- columns_removed
- added_rows
- removed_rows
- old_rows
- new_rows

to understand what actually changed.

Likewise:

features_changed = true

is only a flag.

If feature_details exists, inspect:

- before
- after
- added
- removed

before deciding whether the feature change is relevant.

Your answer must be contextual and evidence-based.
"""

        user_prompt = f"""
Analyze this DATAGIT ML experiment comparison.

The following information was collected directly from the project's
Git, DVC, dataset, code, and ML run evidence.

DO NOT invent anything outside this evidence.

DATAGIT EVIDENCE:

{json.dumps(
    comparison_data,
    indent=2,
    default=str,
)}

Analyze:

1. What actually changed?
2. Which changes are relevant to ML performance?
3. Which changes are irrelevant?
4. How did model performance change?
5. What are the strongest possible contributors?
6. What cannot be determined from the evidence?
7. What should the developer investigate next?

Return ONLY the requested JSON structure.
"""

        try:

            response = client.chat.completions.create(
                model=LLMAnalysisService.MODEL,

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

                temperature=0.2,

                response_format={
                    "type": "json_object"
                },
            )

        except Exception as e:

            raise RuntimeError(
                f"Groq LLM request failed: {str(e)}"
            )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        try:

            result = json.loads(content)

        except json.JSONDecodeError as e:

            raise RuntimeError(
                f"Groq returned invalid JSON: {str(e)}"
            )

        return result