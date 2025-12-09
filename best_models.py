from pathlib import Path
import json

out = Path('outputs')
path_list = []
model_best_coord_error = []
model_best_recall_pos = []
for output_path in out.rglob('*.json'):
    with open(str(output_path), 'r') as f:
        metric = json.load(f)
    if 'reg_error_val' not in metric:
        continue
    path_list.append(output_path)
    if min(metric['reg_error_val']) < 10:
        continue
    model_best_coord_error.append( min(metric['reg_error_val']))
    model_best_recall_pos.append( max(metric['recall_pos']))
idx = list(range(len(model_best_coord_error)))
print('coord error')
idx.sort(key = lambda x: model_best_coord_error[x])
for i in idx[:4]:
    print(str(path_list[i]), f' {model_best_coord_error[i]}')
print('recall pos')
idx.sort(key = lambda x: model_best_recall_pos[x], reverse=True)
for i in idx[:4]:
    print(str(path_list[i]), f' {model_best_recall_pos[i]}')
    