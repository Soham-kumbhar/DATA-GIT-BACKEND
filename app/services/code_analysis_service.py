import re


class CodeAnalysisService:

    @staticmethod
    def analyze_patch(patch: str) -> dict:
        """
        Analyze a Git patch and identify ML-relevant code changes.
        """

        result = {
            "files_changed": [],
            "model_changed": False,
            "features_changed": False,
            "parameters_changed": False,
            "preprocessing_changed": False,
            "training_changed": False,
            "changes": [],
            "feature_details": None,
        }

        if not patch:
            return result

        current_file = None

        for line in patch.splitlines():

            # Detect changed file
            if line.startswith("diff --git"):
                match = re.search(r" b/(.+)$", line)

                if match:
                    current_file = match.group(1)

                    if current_file not in result["files_changed"]:
                        result["files_changed"].append(current_file)

            # Ignore Git metadata lines
            if line.startswith("+++ ") or line.startswith("--- "):
                continue

            # Only analyze additions/removals
            if not (
                line.startswith("+")
                or line.startswith("-")
            ):
                continue

            change_type = (
                "added"
                if line.startswith("+")
                else "removed"
            )

            code = line[1:].strip()

            if not code:
                continue

            # -----------------------------
            # Feature changes
            # -----------------------------
            if (
                "X =" in code
                or "features" in code.lower()
                or "feature" in code.lower()
            ):
                result["features_changed"] = True

                result["changes"].append({
                    "type": change_type,
                    "category": "features",
                    "file": current_file,
                    "code": code,
                })

            # -----------------------------
            # Model changes
            # -----------------------------
            if (
                "Regression(" in code
                or "Classifier(" in code
                or "Regressor(" in code
                or "model =" in code
                or "Pipeline(" in code
            ):
                result["model_changed"] = True

                result["changes"].append({
                    "type": change_type,
                    "category": "model",
                    "file": current_file,
                    "code": code,
                })

            # -----------------------------
            # Parameter changes
            # -----------------------------
            if any(
                keyword in code
                for keyword in [
                    "test_size",
                    "random_state",
                    "max_depth",
                    "n_estimators",
                    "learning_rate",
                    "alpha",
                    "C=",
                ]
            ):
                result["parameters_changed"] = True

                result["changes"].append({
                    "type": change_type,
                    "category": "parameters",
                    "file": current_file,
                    "code": code,
                })

            # -----------------------------
            # Preprocessing changes
            # -----------------------------
            if any(
                keyword in code
                for keyword in [
                    "StandardScaler",
                    "MinMaxScaler",
                    "RobustScaler",
                    "OneHotEncoder",
                    "SimpleImputer",
                    "preprocess",
                ]
            ):
                result["preprocessing_changed"] = True

                result["changes"].append({
                    "type": change_type,
                    "category": "preprocessing",
                    "file": current_file,
                    "code": code,
                })

            # -----------------------------
            # Training changes
            # -----------------------------
            if any(
                keyword in code
                for keyword in [
                    ".fit(",
                    "train_test_split",
                    "cross_val_score",
                    "GridSearchCV",
                    "RandomizedSearchCV",
                ]
            ):
                result["training_changed"] = True

                result["changes"].append({
                    "type": change_type,
                    "category": "training",
                    "file": current_file,
                    "code": code,
                })

        # Extract detailed feature information
        result["feature_details"] = (
            CodeAnalysisService.extract_features(
                result["changes"]
            )
        )

        return result

    @staticmethod
    def extract_features(
        changes: list[dict],
    ) -> dict | None:

        removed = None
        added = None
        file = None

        for change in changes:

            if change["category"] != "features":
                continue

            code = change["code"]

            match = re.search(
                r'X\s*=\s*data\[\[(.*?)\]\]',
                code,
            )

            if not match:
                continue

            raw_features = match.group(1)

            features = re.findall(
                r'"([^"]+)"',
                raw_features,
            )

            file = change["file"]

            if change["type"] == "removed":
                removed = features

            elif change["type"] == "added":
                added = features

        if removed is None and added is None:
            return None

        before = removed or []
        after = added or []

        return {
            "file": file,
            "before": before,
            "after": after,
            "added": [
                feature
                for feature in after
                if feature not in before
            ],
            "removed": [
                feature
                for feature in before
                if feature not in after
            ],
        }