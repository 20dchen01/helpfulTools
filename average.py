import pandas as pd

# read the original Excel file
df = pd.read_excel('training_results.xlsx')

# group the data by Model name and Task, and calculate the mean of Results for each group
grouped = df.groupby(['Model', 'Task'], as_index=False)['Training Results'].mean()

grouped['Training Results'] = grouped['Training Results'] * 100
grouped['Training Results'] = grouped['Training Results'].round(2)

# write the grouped data to a new Excel file
grouped.to_excel('grouped_data.xlsx', index=False)