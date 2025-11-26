# 图像-图像相似度功能使用指南

## 📝 修改概述

已成功修改 `vwsd_clip_baseline.py`，使其支持**图像-图像相似度计算**，同时保持原有的**文本-图像相似度计算**功能。

## 🎯 核心修改

### 1. `embedding_clip.py` 新增方法
- `get_image_embedding(images, batch_size)` - 提取图像嵌入向量
- `get_image_similarity(query_images, candidate_images, batch_size)` - 计算图像间相似度

### 2. `vwsd_clip_baseline.py` 新增参数
- `--use-image-query` - 启用图像-图像模式
- `-i, --image-dir` - 查询图像目录（默认：`image`）
- `--image-pattern` - 图像命名模式（默认：`generated_{n}.jpg`）

## 🚀 使用方法

### 方式1: 原始文本-图像模式
```bash
vwsd-clip-baseline -l en -m laion/CLIP-ViT-L-14-laion2B-s32B-b82K
```
- 使用文本描述与候选图像比较
- 结果保存在 `result/mask.target_word/` 等目录

### 方式2: 新的图像-图像模式
```bash
vwsd-clip-baseline -l en -m laion/CLIP-ViT-L-14-laion2B-s32B-b82K \
    --use-image-query \
    -i image \
    --image-pattern "generated_{n}.jpg"
```
- 使用查询图像与候选图像比较
- 结果保存在 `result/image_to_image_similarity/`

## 📊 评估结果

使用相同的评估命令：
```bash
# 文本-图像模式评估
vwsd-ranking-metric -p result/mask.target_word -r all.gold.en.tmp

# 图像-图像模式评估
vwsd-ranking-metric -p result/image_to_image_similarity -r all.gold.en.tmp
```

输出指标相同：MRR、Recall@1、Recall@3 等

## 🎨 图像命名规则示例

### 默认命名（推荐）
```
image/
  ├── generated_0.jpg
  ├── generated_1.jpg
  ├── generated_2.jpg
  └── ...
```
使用：`--image-pattern "generated_{n}.jpg"`

### 其他命名方式
```bash
# 如果命名为 query_0.png, query_1.png, ...
--image-pattern "query_{n}.png"

# 如果命名为 img_000.jpg, img_001.jpg, ...（带前导零）
--image-pattern "img_{n:03d}.jpg"

# 如果命名为 sample_0001.jpg, sample_0002.jpg, ...
--image-pattern "sample_{n:04d}.jpg"
```

## 📁 输出文件结构

### 图像-图像模式输出
```
result/
  └── image_to_image_similarity/
      ├── prediction.en.txt        # 预测结果（用于评估）
      └── full_result.en.csv       # 完整结果（含相似度分数）
```

### 文本-图像模式输出（保持不变）
```
result/
  ├── mask.target_word/
  │   ├── prediction.en.txt
  │   └── full_result.en.csv
  ├── mask.target_phrase/
  │   └── ...
  └── ...
```

## ⚠️ 注意事项

1. **语言支持**：图像-图像模式目前只支持英语（`-l en`），因为需要使用 CLIP 模型
2. **图像编号**：图像编号从 0 开始，对应数据集样本顺序
3. **缺失图像**：如果某个图像文件不存在，该样本会被跳过并记录警告
4. **模型一致性**：建议使用 `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` 以获得最佳性能

## 💡 使用场景

### 场景1: 生成图像评估
你使用 Stable Diffusion 等模型为每个 V-WSD 样本生成了一张图像：
```bash
vwsd-clip-baseline -l en -m laion/CLIP-ViT-L-14-laion2B-s32B-b82K \
    --use-image-query \
    -i generated_images \
    --image-pattern "sd_output_{n}.png"
```

### 场景2: 参考图像检索
你有一组参考图像，想找到最相似的候选图像：
```bash
vwsd-clip-baseline -l en -m laion/CLIP-ViT-L-14-laion2B-s32B-b82K \
    --use-image-query \
    -i reference_images \
    --image-pattern "ref_{n}.jpg"
```

### 场景3: 方法对比
同时运行两种模式，对比文本检索和图像检索的效果：
```bash
# 运行文本-图像模式
vwsd-clip-baseline -l en -m laion/CLIP-ViT-L-14-laion2B-s32B-b82K

# 运行图像-图像模式
vwsd-clip-baseline -l en -m laion/CLIP-ViT-L-14-laion2B-s32B-b82K --use-image-query

# 对比评估结果
vwsd-ranking-metric -p result/mask.target_word -r all.gold.en.tmp
vwsd-ranking-metric -p result/image_to_image_similarity -r all.gold.en.tmp
```

## 🔧 完整工作流程示例

```bash
# 1. 安装项目（如果还没安装）
cd vwsd_experiment
pip install -e .

# 2. 准备查询图像（放在 image/ 文件夹中）
# 确保图像命名为 generated_0.jpg, generated_1.jpg, ...

# 3. 运行图像-图像相似度计算
vwsd-clip-baseline -l en -m laion/CLIP-ViT-L-14-laion2B-s32B-b82K \
    --use-image-query \
    -i image \
    --image-pattern "generated_{n}.jpg"

# 4. 准备 gold 标签文件
python -c "
with open('dataset/label/en.test.gold.v1.1.txt', 'r') as f:
    with open('all.gold.en.tmp', 'w') as out:
        for line in f:
            out.write(line.strip() + '\ten\n')
"

# 5. 评估结果
vwsd-ranking-metric -p result/image_to_image_similarity -r all.gold.en.tmp

# 6. 查看结果
cat result/image_to_image_similarity/prediction.en.txt
```

## 📈 预期输出

评估命令会输出类似以下的指标：
```
MRR: 0.XX
Recall@1: 0.XX
Recall@3: 0.XX
Recall@5: 0.XX
Recall@10: 0.XX
```

完整结果 CSV 文件包含：
- `language`: 语言代码
- `data`: 样本编号
- `candidate`: 排序后的候选图像列表
- `relevance`: 相似度分数列表
- `query_image`: 查询图像文件名
- `input_type`: 'image_query'
- `prompt`: 'image_to_image'

## 🎓 技术细节

### 相似度计算
- 使用 CLIP 提取图像特征
- 计算余弦相似度
- 分数范围：0-100（越高越相似）

### 批处理
- 支持 `--batch-size` 参数控制批大小
- 默认自动批处理以优化内存使用

### 可视化
- 添加 `--plot` 参数可生成相似度可视化
- 保存在 `result/visualization/en/` 目录
