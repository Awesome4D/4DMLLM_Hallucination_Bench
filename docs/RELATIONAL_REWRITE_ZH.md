# 2026-09-08：关系理解重写与新增数据分析

论文标题：**Do LiDAR Language Models Really Understand Spatio-temporal Relationships?**

本轮不产生新的神经模型回答，不改写原始问题和预测。新增的是从原有数据编译的关系对照子集、逐谓词分析，以及对 TSCD 的精确代数分析。

## 1. 可复现的新结果

- NoMeta：348 对不复用问题的跨场景二分类对，完整 prompt 和序列长度完全相同，真值相反。包括 104 对存在性、41 对位置、203 对运动；696 题、140 个场景。匹配只依赖问题、标签、场景和长度，不读取模型正确性。
- NoMeta baseline：43/348 对两边同时答对，即 12.36%；单题准确率 53.30%。285 对给出相同答案，20 对两边都错。
- TSCD 的联合准确率依次为 12.07%、12.07%、11.49%，没有提高 baseline 的 12.36%。
- Meta 的完整 prompt 包含自车运动文本，因而匹配出另一组更小的 38 对。baseline 2/38，TSCD 4/38。不能把两个不一致子集当作公平的跨模型提升比较。
- 722 道左右运动题包括 302 个正例。两个配置、全部五种设置中的 7,220 条回答全为 No，正类召回为零。

## 2. 理论解释的范围

令 clean logits 为 b+e_c、shuffle logits 为 b+e_s。TSCD=(1+alpha)z_c-alpha*z_s=b+(1+alpha)e_c-alpha*e_s。因此共同分量 b 并未从最终 logits 消失。

这是一条代数恒等式，不是从保存结果中估计出了模型内部的唯一偏置分解。若两个分支仅差一个对所有 token 相同的常数，softmax 分布完全不变。我们没有保留 logits 或 attention map，不能宣称实测了 attention 机制，也不能宣称消除了 bias。

完整问题相同、真值相反的二分类对中，确定性的有效纯文本预测器单题准确率为 50%，联合正确率为零。独立随机预测可获得非零联合正确率，p(1-p)<=25%，所以联合正确率非零本身也不是证据使用的充分证明。

## 3. 运行

```bash
python -m pip install -e '.[test]'
LIDAR_HALLU_DATA=data python scripts/relational_audit/test_contracts.py
OPENBLAS_NUM_THREADS=1 python scripts/relational_audit/analyze_rewrite.py --data data --output results/relational_rewrite
python scripts/relational_audit/analyze_prompt_matched_pairs.py --data data --output results/relational_rewrite
```

`results/relational_rewrite/prompt_matched_pairs_*.jsonl` 是新增对照子集的完整成员表。
`prompt_matched_results.csv` 和 `motion_predicates.csv` 对应论文新增结果。
工作流会从已提交的完整原始数据重新生成并提交这些文件。

## 4. 不可混淆的边界

这些是现有场景的事后自然跨场景对照，不是新采样的独立测试集，不是只修改一个物体的因果干预。场景间仍有未匹配因素，原推理仍有采样随机性。不能按独立对计算二项式置信区间，也不能把 348 对当成新增 696 个独立原始问题。

数据标准为对象、时间、参照系、关系规则的显式约定。标准答案可追溯不保证模型传感输入中证据充分。当前时间候选的长度捷径保留并公开诊断，不把旧数据重新命名为已经去偏的新基准。
