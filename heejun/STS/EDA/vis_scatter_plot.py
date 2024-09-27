import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target_folder')
args = parser.parse_args()

sample_submission_550 = pd.read_csv('../data/sample_submission_550.csv')

save_path = f'./lightning_logs/{args.target_folder}'
valid = pd.read_csv('../../data/dev.csv')
inference_valid = pd.read_csv(f'{save_path}/output_valid-{args.target_folder}.csv')
valid['preds'] = inference_valid['target']

target = valid['label']
preds = valid['preds']

# save valid
valid.to_csv(f'{save_path}/valid_target_preds.csv', index=False)

plt.scatter(target, preds)
plt.xlabel('target')
plt.ylabel('preds')
# draw y=x line
plt.plot([0, 5], [0, 5], color='red')

pearson = args.target_folder.split('_')[0]
model_name = args.target_folder.split('_')[1]
plt.title(f'Pearson: <{pearson}>')
plt.suptitle(f'Model: <{model_name}>')
plt.savefig(f'{save_path}/valid_corr.png')