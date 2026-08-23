import papermill as pm

seeds = [43, 44]
# emotion | ag_news | banking77 
datasets = ["emotion","banking77"]

for dataset in datasets:
    for seed in seeds:
        print(f"Running notebook for {dataset} {seed}")
        pm.execute_notebook(
            input_path="local_llm_qlora.ipynb",
    output_path=f"{seed}_{dataset}_llm_qlora.ipynb",
        parameters={
            "seed": seed,
            "dataset": dataset
        }
    )

print("Finished!")