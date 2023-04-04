""" Hunter's code but adapted for baseline models"""
import os
import json
import statistics
import matplotlib.pyplot as plt
import numpy as np

# path to the main directory
directory = '../../768DistilbertL1012/Totest/6-768-baseline'
dic = {}

# Iterate through all output subfolder to read all results
for seed in os.listdir(directory):
    for tasksname in os.listdir(directory + '/' + seed):
        if tasksname == 'wnli': # skip wnli
            continue
        with open(directory + '/' + seed + '/' + tasksname + '/' + 'eval_results.json', 'r') as file:
            contents = json.load(file)
            # get the value of the desired line from the json contents
            if tasksname == 'cola':
                value = contents['eval_matthews_correlation']
            elif tasksname == 'stsb':
                value = contents['eval_spearmanr']
            elif tasksname == 'qqp' or tasksname == 'mrpc':
                value = contents['eval_f1']
            else:
                value = contents["eval_accuracy"]
            
            # check empty for dic
            if dic.get(tasksname) is not None:
                dic[tasksname].append(value)
            else:
                dic[tasksname] = []
                dic[tasksname].append(value)

#iterate through bert folder
directory2 = '../../768DistilbertL1012/BERT-BASE-UNCASED'
dic2 = {}
for seed in os.listdir(directory2):
    for tasksname in os.listdir(directory2 + '/' + seed):
        if tasksname == 'wnli': # skip wnli
            continue
        with open(directory2 + '/' + seed + '/' + tasksname + '/' + 'eval_results.json', 'r') as file2:
            contents = json.load(file2)
            # get the value of the desired line from the json contents
            if tasksname == 'cola':
                value = contents['eval_matthews_correlation']
            elif tasksname == 'stsb':
                value = contents['eval_spearmanr']
            elif tasksname == 'qqp' or tasksname == 'mrpc':
                value = contents['eval_f1']
            else:
                value = contents["eval_accuracy"]
            
            # check empty for dic
            if dic2.get(tasksname) is not None:
                dic2[tasksname].append(value)
            else:
                dic2[tasksname] = []
                dic2[tasksname].append(value)



tasks = ['qnli', 'mrpc', 'rte','cola','mnli','qqp','stsb','sst2']
#tasks = ['qnli', 'mrpc', 'rte','cola','mnli','qqp','stsb']
data = {task: None for task in tasks}
data2 = {task: None for task in tasks}

for k,v in dic.items():
    mean = statistics.mean(v)
    std = statistics.stdev(v)
    error = std / (len(v)**0.5)
    data[k] = {'mean': mean, 'error':error}
    print(f"{mean} ", end='')
print("")

for k,v in dic2.items():
    mean = statistics.mean(v)
    std = statistics.stdev(v)
    error = std / (len(v)**0.5)
    data2[k] = {'mean': mean, 'error':error}
    print(f"{mean} ", end='')
print("")

# Result from the official bert
# bert = {
#     'qnli': {'mean': 0.8931, 'std': 0.0046, 'error': 0.0021},
#     'mrpc': {'mean': 0.8627, 'std': 0.0156, 'error': 0.0070},
#     'rte': {'mean': 0.7143, 'std': 0.0256, 'error': 0.0115},
#     'cola': {'mean': 0.4887, 'std': 0.0252, 'error': 0.0113},
#     'mnli': {'mean': 0.8483, 'std': 0.0022, 'error': 0.0010},
#     'qqp': {'mean': 0.8766, 'std': 0.0013, 'error': 0.0006},
#     'stsb': {'mean': 0.9104, 'std': 0.0046, 'error': 0.0021},
#     'sst2': {'mean': 0.9174, 'std': 0.0045, 'error': 0.0020}
# }

# extract the data into separate lists
keys = list(data.keys())
means = [data[key]['mean'] for key in keys]

errors = [data[key]['error'] for key in keys]
#bmeans = [bert[key]['mean'] for key in keys]
means2 = [data2[key]['mean'] for key in keys]
errors2 = [data2[key]['error'] for key in keys]

# customize the error bars
error_caps = dict(lw=1, capsize=5, capthick=1)
error_kwargs = dict(ecolor='black', lw=1, alpha=0.7, capsize=5, capthick=1)


# plot the data
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(np.arange(len(keys)), means2, yerr=errors2,align='center', alpha=0.5, error_kw=error_kwargs, capsize=3, color='k')
ax.bar(np.arange(len(keys)), means, yerr=errors, align='center', alpha=0.5, error_kw=error_kwargs, capsize=3, color='b')
ax.set_xticks(np.arange(len(keys)))
ax.set_xticklabels(keys)
ax.set_ylabel('Mean')
ax.set_title('Evaluation Scores : 6-768-baseline vs BERT-BASE-UNCASED')
plt.savefig('fig/6-768-baseline.png', format='png')
plt.xlabel('GLUE tasks')
plt.show()