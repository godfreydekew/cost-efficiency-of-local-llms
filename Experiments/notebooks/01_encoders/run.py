import papermill as pm

seeds = [42, 43, 44]
# emotion | ag_news | banking77 
datasets = ["emotion", "ag_news", "banking77"]

for dataset in datasets:
    for seed in seeds:
        print(f"Running notebook for {dataset} {seed}")
        pm.execute_notebook(
            input_path="distilbert_tuning.ipynb",
    output_path=f"{seed}_{dataset}_robert.ipynb",
        parameters={
            "seed": seed,
            "dataset": dataset
        }
    )

print("Finished!")