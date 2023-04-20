import os
import pandas as pd

# Read the Excel file
input_file = 'training_results.xlsx'
df = pd.read_excel(input_file)

# Get unique model names
models = df['Model'].unique()

# Create a dictionary to map tasks to their new names
task_mapping = {
    'qnli': 'qnli (A)',
    'mnli': 'mnli (A)',
    'rte': 'rte (A)',
    'sst2': 'sst2 (A)',
    'qqp': 'qqp (F)',
    'mrpc': 'mrpc (F)',
    'stsb': 'stsb (P)',
    'cola': 'cola (M)'
}

# Create a directory to store the CSV files
os.makedirs('csvs', exist_ok=True)

for model in models:
    # Filter the original DataFrame by the model name
    model_df = df[df['Model'] == model]
    
    # Drop the 'Model' column
    model_df = model_df.drop(columns=['Model'])
    
    # Rename the 'Task' column values using the task_mapping dictionary
    model_df['Task'] = model_df['Task'].apply(lambda task: task_mapping[task])
    
    # Pivot the DataFrame to have the desired format
    model_df = model_df.pivot_table(index='Seed', columns='Task', values='Training Results')
    
    # Reset the index
    model_df.reset_index(inplace=True)
    
    # Save the new DataFrame as a CSV file under the 'csvs' folder
    model_df.to_csv(f'csvs/{model}.csv', index=False)
