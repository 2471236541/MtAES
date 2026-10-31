# 上传目录说明

本目录是 Prompt 7 论文实验的源码包。GitHub 仓库创建和上传由作者自行操作。

## 保留的文件

| 文件或目录 | 用途 |
| --- | --- |
| `README.md` | 英文项目介绍、数据获取和执行顺序 |
| `requirements.txt` | 当前训练环境中核对过的 Python 库版本 |
| `.gitignore` | 排除运行后产生的数据、权重、缓存、日志及本地凭证 |
| `configs/prompt7.json` | 根据代码和实际划分文件汇总的参数记录 |
| `configs/prompt7_split_ids.json` | 五折外层、三折内层及 refit 的作文 ID 划分，不含文本或预测 |
| `trait_questions_prompt7/` | ideas、organization、style、conventions 的问题列表 |
| `run_prompt7_evaluation.py`、`trait_analysis_pipeline.py` | Qwen 初始分析与批量评分；完整 prompt 在后一个文件中 |
| `prompt7_mapper.py` | 初始维度映射和 overall 合成 |
| `build_nested_outer_folds.py`、`build_nested_inner_kfold.py` | 外层及内层划分 |
| `train_sidecar_bert.py`、`infer_sidecar_bert.py` | 候选模型训练和推理 |
| `sidecar_text_preset.py` | 候选模型输入构造 |
| `run_v10_nested_pipeline.sh` | 初始候选分类器训练入口 |
| `run_e59_outer_sealed_once.sh` | 基础修正阶段入口 |
| `run_e90_outer_sealed_once.sh` | 最终双候选 ideas 修正入口 |
| `apply_e90_e79_e87_outer.py` | 方向一致性规则与结果汇总 |
| 其余根目录脚本 | 上述入口使用的合并、门控、评估与检查依赖 |
| `docs/REPRODUCTION.md` | 分阶段运行命令和路径说明 |
| `docs/source_manifest.json` | 复制文件的 SHA-256 校验值和配置来源 |

## 整理范围

复制的实验代码和四份问题 JSON 与原项目逐字节一致，没有改模型、参数、路径或公式。
新增内容是 README、依赖清单、配置记录、说明文档和 `.gitignore`。
参数 JSON 用于查阅；原脚本不会自动加载它。

未放入绘图代码、图片、逐条预测结果、原始作文、缓存、模型权重、日志、论文及旧 Git 历史。
原项目仍保留全部实验材料。本目录没有执行模型训练或 GitHub 上传。

## 运行前需注意的原有路径

按此次约定，原代码的绝对路径没有改动。主要有以下几处：

1. `run_v10_nested_pipeline.sh` 第 12 行的 `PY` 是原服务器环境路径，换机器运行时需改成实际 Python 路径。
2. `run_e59_outer_sealed_once.sh` 和 `run_e90_outer_sealed_once.sh` 支持通过环境变量指定 `ROOT`、`PY` 和输入输出目录，文档已给出命令。
3. `audit_e90_outer_cleanliness.py` 中的 `ROOT` 固定指向原项目，执行前需检查。

不要直接在原项目根目录执行 `git add .`。上传的是本目录中的文件，不包含原项目的其他内容。
