from app.services.ll_analysis_service import LLMAnalysisService


comparison_data = {
    "run_1": 1,
    "run_2": 2,

    "changes": {
        "dataset_changed": True,
        "model_changed": False,
        "features_changed": True,
        "parameters_changed": False,
        "code_changed": True,
    },

    "metrics": {
        "r2": {
            "before": 0.94,
            "after": 0.82,
            "change": -0.12,
            "percentage_change": -12.77,
        },
        "rmse": {
            "before": 42000,
            "after": 65000,
            "change": 23000,
            "percentage_change": 54.76,
        },
    },

    "dataset_details": {
        "dataset": "data/cars.csv",
        "dataset_changed": True,
        "old_rows": 10,
        "new_rows": 11,
        "columns_added": [],
        "columns_removed": [],
        "rows_added": 1,
        "rows_removed": 0,
        "added_rows": [
            {
                "year": 2022,
                "km_driven": 5000,
                "engine_cc": 1500,
                "price": 1050000,
                "age": 4,
            }
        ],
    },

    "code_details": {
        "files_changed": [
            "data/cars.csv",
            "src/train.py",
        ],
        "model_changed": False,
        "features_changed": True,
        "parameters_changed": False,
        "preprocessing_changed": False,
        "training_changed": False,

        "changes": [
            {
                "type": "removed",
                "category": "features",
                "file": "src/train.py",
                "code": 'X = data[["year", "km_driven", "engine_cc"]]',
            },
            {
                "type": "added",
                "category": "features",
                "file": "src/train.py",
                "code": 'X = data[["year", "km_driven", "engine_cc", "age"]]',
            },
        ],

        "feature_details": {
            "file": "src/train.py",
            "before": [
                "year",
                "km_driven",
                "engine_cc",
            ],
            "after": [
                "year",
                "km_driven",
                "engine_cc",
                "age",
            ],
            "added": ["age"],
            "removed": [],
        },
    },
}


print("=" * 60)
print("LLM ANALYSIS")
print("=" * 60)

result = LLMAnalysisService.analyze_comparison(
    comparison_data
)

print(result)