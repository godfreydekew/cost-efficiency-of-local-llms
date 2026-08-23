import papermill as pm

seeds = [42]
# emotion | ag_news | banking77 
datasets = ["emotion","banking77", "20_newsgroups"]

for dataset in datasets:
    for seed in seeds:
        print(f"Running notebook for {dataset} {seed}")
        pm.execute_notebook(
            input_path="api_llm_zero_fewshot.ipynb",
    output_path=f"{seed}_{dataset}_llm_api.ipynb",
        parameters={
            "seed": seed,
            "dataset": dataset
        }
    )

print("Finished!")