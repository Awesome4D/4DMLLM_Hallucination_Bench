# nuScenes LiDAR 下载与本项目数据对应

核对日期：2026-09-08。本文只准备原始传感数据与标注，不下载模型检查点。

## 1. 先区分三种用途

| 用途 | 需要什么 | 现在能否直接做 |
|---|---|---|
| 重算现有论文表格、误报漏报和先验分析 | 仓库内问题、保存的预测、CPU 分析代码 | 可以，无需 nuScenes、GPU 或 checkpoint |
| 查看真实 LiDAR 样本、按场景回溯 GT、制作真实 teaser | nuScenes trainval metadata、对应 LiDAR 文件；配图时可选同步 RGB | 需另行下载原始数据 |
| 生成新的 B4DL 回答 | 原始运行环境、LiDAR 特征缓存、base model、projector、checkpoint | 本轮不恢复检查点；不能仅凭 JSON 和原始 LiDAR 直接调用现有推理脚本 |

历史推理读取的是 `stage2_features/<scene_id>.npy`，不是在线从 `.bin` 提取特征。已有缓存缺失时，需要原 LiDAR encoder 权重及一致的预处理配置重新提取。下载 raw LiDAR 不会自动恢复历史缓存。

## 2. 官方下载入口和版本

- [nuScenes 官方下载页](https://www.nuscenes.org/nuscenes#download)
- [nuScenes devkit 官方安装与目录说明](https://github.com/nutonomy/nuscenes-devkit#nuscenes-setup)
- [nuScenes 官方数据结构教程](https://www.nuscenes.org/public/tutorials/nuscenes_tutorial.html)

登录自己的 nuScenes 账号并接受数据使用条款。从完整数据集选择 **v1.0-trainval**，不是 v1.0-test，也不能用 mini 替代论文的 150 个场景。我们的 JSON 中 `split: test` 是本项目评测角色，不代表官方 nuScenes test split。

下载两部分：

1. **Trainval metadata**，包含物体标注、自车位姿、标定、sample 和 scene 索引。页面通常提供 metadata 压缩包，常见名称为 `v1.0-trainval_meta.tgz`。以账号内当前文件列表为准。
2. **Trainval 的 LiDAR sensor/blob 分片**。页面若提供 LiDAR-only 下载，优先选择它，并取齐相关分片。若只提供完整 sensor 分片，原始 LiDAR 仍在其中。不要假定一个分片恰好对应全部 validation 场景。

必须保留 metadata，不能只下载 `.bin`。仅做 LiDAR 分析不需要全部 RGB 和 radar；需要真实 RGB+LiDAR 对照 teaser 时，另取对应的 `CAM_FRONT` 或所需相机文件。现有框级 QA 规则不要求 lidarseg、panoptic 或 CAN bus 扩展。

官方页面依赖 JavaScript 和账号授权。这里不编造免登录的 trainval 直链、不提供登录令牌，也不写未经验证的实时总大小。

## 3. 目录和解压

建议把大文件放在共享存储，而不是 Git 工作树：

```bash
export NUSCENES_ROOT=/group/your_group/datasets/nuscenes
export NUSCENES_ARCHIVES=/group/your_group/downloads/nuscenes
mkdir -p "$NUSCENES_ROOT"

# 在 Linux/Bash 中，将实际下载的 trainval 压缩包合并解压到同一个根目录。
find "$NUSCENES_ARCHIVES" -maxdepth 1 -type f \
  \( -name 'v1.0-trainval*.tgz' -o -name 'v1.0-trainval*.tar.gz' \) -print0 |
while IFS= read -r -d '' archive; do
  tar -xf "$archive" -C "$NUSCENES_ROOT" || exit 1
done
```

路径是示例，需要替换为自己有写权限的目录。保留同名目录的内容并合并，不要先删除已解压分片。确认没有形成多余的嵌套 `nuscenes/nuscenes/`：

```text
$NUSCENES_ROOT/
  v1.0-trainval/
    scene.json
    sample.json
    sample_data.json
    sample_annotation.json
    ego_pose.json
    calibrated_sensor.json
    sensor.json
    instance.json
    category.json
    ...其他官方 metadata 表...
  samples/
    LIDAR_TOP/                 # 有标注的关键帧
    CAM_FRONT/                # 可选，只在需要 RGB 对照时取
  sweeps/
    LIDAR_TOP/                 # 中间扫描，完整重建/多 sweep 配置时需要
  maps/                       # metadata 包提供的基础地图，保留
```

标准完整安装遵循官方 devkit，包含相应的 samples/sweeps/maps。对于本仓库的关键帧检查器，`samples/LIDAR_TOP` 和 metadata 足够。多 sweep 融合及精确恢复历史特征时，必须先核对原提取配置，不能擅自认为 sweeps 可省略。

## 4. 只定位本 benchmark 的 150 个场景

仓库已经提供 `data/manifests/benchmark_scenes.json`，含 benchmark scene ID、官方 scene token，以及题目中保留的 frame-to-sample-token 对应。

只有 metadata 时，先生成具体文件清单：

```bash
python scripts/check_nuscenes_assets.py \
  --dataroot "$NUSCENES_ROOT" --manifest-only
```

输出为 `data/local/nuscenes_assets.json` 和 `data/local/nuscenes_assets.files.txt`。前者包含每帧对应的官方 scene name、sample token、时间戳、LiDAR 文件名。`MANIFEST_ONLY` 只表示索引建立成功，不表示点云已经下载。

下载之后去掉该参数，检查文件是否存在且非空：

```bash
python scripts/check_nuscenes_assets.py --dataroot "$NUSCENES_ROOT"

# 需要中间 LiDAR sweeps 时：
python scripts/check_nuscenes_assets.py --dataroot "$NUSCENES_ROOT" --include-sweeps

# 需要真实 RGB+LiDAR teaser 时：
python scripts/check_nuscenes_assets.py \
  --dataroot "$NUSCENES_ROOT" --camera-channels CAM_FRONT
```

检查器不依赖 checkpoint 或 nuScenes SDK。它依据 scene 链恢复关键帧顺序，并核对现有题目的 frame/sample_token 对应。若出现 `B4DL frame-order mismatch`，必须恢复原始 B4DL metadata，不能静默猜测帧号。不要把 B4DL 的数字 scene ID 直接拼成 nuScenes 文件名。

从已有完整共享数据目录复制这些文件到子集目录，可使用：

```bash
# 先用 --manifest-only 生成所需列表；metadata 仍需完整复制或共享。
rsync -av --files-from=data/local/nuscenes_assets.files.txt \
  "$NUSCENES_ROOT/" /path/to/nuscenes_subset/
rsync -av "$NUSCENES_ROOT/v1.0-trainval/" \
  /path/to/nuscenes_subset/v1.0-trainval/
```

这只是从已有本地/挂载数据复制子集，不是绕过 nuScenes 登录下载。

## 5. 当前核验范围

数据导入脚本已在用户上传的真实 benchmark/prediction ZIP 上运行并通过逐条匹配。原始 nuScenes 检查器目前有合成 metadata 的单元测试，**本轮未下载 trainval 原始点云，未在真实完整 nuScenes 上运行**。其 PASS 仅指 metadata 对齐和文件存在，不代表已校验点云内容、重建 LiDARCLIP 特征或运行模型。

Checkpoint 暂不处理。后续新推理还需完整 B4DL/VTimeLLM、遗留代码导入的 `evaluate_simple_tasks.py`、feature cache 或 encoder 权重、base model、projector 和两个训练 checkpoint。见 [MIGRATION.md](MIGRATION.md)。

原始 nuScenes 数据遵循官方条款，不上传到 Git；本次没有改变仓库的私有状态，也没有替上游重新授予许可。
