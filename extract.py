import os
import pandas as pd

# define the root directory of the models
root_dir = '../../768DistilbertL1012/Totest'

# initialize an empty dataframe
df = pd.DataFrame(columns=['Model', 'Seed', 'Task', 'Training Results'])

# iterate over each model folder
for model in os.listdir(root_dir):
    if not os.path.isdir(os.path.join(root_dir, model)):
        continue
    
    # iterate over each seed folder within the model folder
    for seed in os.listdir(os.path.join(root_dir, model)):
        if not os.path.isdir(os.path.join(root_dir, model, seed)):
            continue
        
        # iterate over each task folder within the seed folder
        for task in os.listdir(os.path.join(root_dir, model, seed)):
            if not os.path.isdir(os.path.join(root_dir, model, seed, task)):
                continue
            
            # read the contents of the readme file and extract the most recent training results
            training_results = ""
            with open(os.path.join(root_dir, model, seed, task, 'README.md'), 'r') as f:
                readme_contents = f.read()
                start_index = readme_contents.find("### Training results")
                if start_index != -1:
                    end_index = readme_contents.find("### Framework versions", start_index + 1)
                    training_results = readme_contents[start_index:end_index].strip().split("\n")[-1]
                    task_name = task.split("_")[0]
                    if task_name in ["cola", "rte", "sst2", "qnli", "mnli"]:
                        training_results = training_results.split()[-2]
                    elif task_name in ["stsb", "mrpc", "qqp"]:
                        training_results = training_results.split()[-4]
                
            # append the training results to the dataframe
            df = df.append({'Model': model, 'Seed': seed, 'Task': task, 'Training Results': training_results}, ignore_index=True)

# save the dataframe to an Excel file
df.to_excel('training_results.xlsx', index=False)