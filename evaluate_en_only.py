#!/usr/bin/env python
"""评估脚本 - 只处理英语数据"""
import sys
import json
import pandas as pd
from statistics import mean
from ranx import Qrels, Run, evaluate

def get_metric(prediction, gold_ranking):
    """计算 MRR 和 Hit Rate"""
    assert all(len(set(p)) == len(p) for p in prediction), prediction
    assert all(r in p for p, r in zip(prediction, gold_ranking)), list(zip(prediction, gold_ranking))
    mrr = mean(1/(p.index(r) + 1) for p, r in zip(prediction, gold_ranking))
    hit = mean(p.index(r) == 0 for p, r in zip(prediction, gold_ranking))
    return mrr, hit

# 参数
prediction_dir = 'result/image_to_image_similarity'
reference_file = 'all.gold.en.tmp'
output_file = 'rank_metrics.jsonl'
metrics_list = ["hit_rate@1", "map@5", "mrr@5", "ndcg@5", "map@10", "mrr@10", "ndcg@10"]

# 读取 reference（只有英语）
with open(reference_file, 'r', encoding='utf-8') as f:
    reference_lines = [i.split('\t') for i in f.read().strip().split("\n") if len(i) > 0]
reference = [x for x, y in reference_lines if y == 'en']

print(f"✓ Reference: {len(reference)} 个英语样本")

# 读取 prediction（只有英语）
prediction_file = f'{prediction_dir}/prediction.en.txt'
with open(prediction_file, 'r', encoding='utf-8') as f:
    prediction_lines = [i.split('\t') for i in f.read().strip().split("\n") if len(i) > 0]

print(f"✓ Prediction: {len(prediction_lines)} 个预测")

# 确保数量匹配
if len(prediction_lines) != len(reference):
    print(f"⚠️  警告: prediction ({len(prediction_lines)}) 和 reference ({len(reference)}) 数量不匹配")
    sys.exit(1)

# 计算指标
metric_dict = {'model': prediction_dir}

# 官方 MRR 和 Hit Rate
m, h = get_metric(prediction_lines, reference)
metric_dict['mrr_official/en'] = m
metric_dict['hit_official/en'] = h

print(f"\n官方指标:")
print(f"  MRR: {m:.4f}")
print(f"  Hit Rate: {h:.4f}")

# ranx 指标
qrels_dict = {str(n): {r: 1} for n, r in enumerate(reference)}
run_dict = {str(n): {c: 1/(1 + r) for r, c in enumerate(x)} for n, x in enumerate(prediction_lines)}

ranx_metrics = evaluate(Qrels(qrels_dict), Run(run_dict), metrics=metrics_list)
metric_dict.update({f"{k}/en": v for k, v in ranx_metrics.items()})

print(f"\nRanx 指标:")
for k, v in ranx_metrics.items():
    print(f"  {k}: {v:.4f}")

# 保存结果
import os
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        metric_all = [json.loads(i) for i in f.read().split("\n") if len(i) > 0]
else:
    metric_all = []

metric_all.append(metric_dict)

with open(output_file, 'w') as f:
    f.write("\n".join([json.dumps(i) for i in metric_all]))

# 显示结果表格
print(f"\n结果已保存到: {output_file}")
print("\n" + "=" * 80)
print(pd.DataFrame(metric_all).to_markdown(index=False))
