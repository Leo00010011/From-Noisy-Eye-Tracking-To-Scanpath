from pathlib import Path
import json

out = Path('outputs')
path_list = []
model_best_coord_error = []
model_best_dur_error = []
model_best_dur_with_good_recall = []
for output_path in out.rglob('*.json'):
    try:
        with open(output_path, 'r') as f:
            metric = json.load(f)
    except Exception:
        continue
    if 'reg_error_val' not in metric:
        continue
    path_list.append(output_path)
    if min(metric['reg_error_val']) < 10:
        continue
    idx = list(range(len(metric['reg_error_val'])))
    min_idx = min(idx, key = lambda x: metric['reg_error_val'][x])
    model_best_coord_error.append( (metric['reg_error_val'][min_idx], 
                                    metric['duration_error_val'][min_idx],
                                    metric['recall_pos'][min_idx],
                                    metric['precision_pos'][min_idx]))
    min_idx = min(idx, key = lambda x: metric['duration_error_val'][x])
    model_best_dur_error.append( (metric['reg_error_val'][min_idx], 
                                    metric['duration_error_val'][min_idx],
                                    metric['recall_pos'][min_idx],
                                    metric['precision_pos'][min_idx]))
    if metric['recall_pos'][min_idx] > 0.30:
        model_best_dur_with_good_recall.append( (metric['reg_error_val'][min_idx], 
                                        metric['duration_error_val'][min_idx],
                                        metric['recall_pos'][min_idx],
                                        metric['precision_pos'][min_idx]))
print(model_best_coord_error)
idx = list(range(len(model_best_coord_error)))
print('coord error')
idx.sort(key = lambda x: model_best_coord_error[x])
for i in idx[:4]:
    print(str(path_list[i]), f' {model_best_coord_error[i]}')
print('duration pos')
idx.sort(key = lambda x: model_best_dur_error[x])
for i in idx[:4]:
    print(str(path_list[i]), f' {model_best_dur_error[i]}')
if model_best_dur_with_good_recall:
    print('duration with good recall')
    idx.sort(key = lambda x: model_best_dur_with_good_recall[x])
    for i in idx[:4]:
        print(str(path_list[i]), f' {model_best_dur_with_good_recall[i]}')
else:
    print('no model with good recall')