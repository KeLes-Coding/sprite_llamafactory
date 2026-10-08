# v3.4.0 定版记录（2026-10-08）

Git 标签：本仓库 `v3.4.0-baseline`；数据生成器 `data_gen/261005` 本地仓库 `v3.4.0`。

## 1. 数据

| 项 | 值 |
|---|---|
| 导出包 | `data/v3-4-0-74a63ca3`（生成端 `data_gen/261005/runs/export_v340/v3-4-0-74a63ca3`） |
| 规模 | train 75,702 行 + validation_1k 1,000 行；L1 15,526，L2–L10 各约 6.1k–7.6k |
| 来源 run | 原子池 `runs/pool-1007_152342`（中文 8,471 原子）→ 拼装 `runs/v340_zh-1007_180008`、`runs/v340_en-1007_180524` |
| seed / 增强 | seed=340；`augmentation: none`（未启用工具名/参数名掩码与缺工具反例，工具展示顺序固定） |

SHA256（`SHA256SUMS`）：

```text
74a63ca394ebe5af08adfe0c2cdc4fcbe442bbf4b163674739ec0bcee6df60f1  action_sequence_sft_train.jsonl
165e6e27edc0d4323890c2b1e4254ccd9ee9c1ec9986ee31a5e0539adaca94b9  action_sequence_sft_validation_1k.jsonl
d41338fbf362a36205c1a1e70d5480881bcabbc69a1020b2cc9cf3572909c578  dataset_info.json
2c1b90af16410b7dede4e260c9e8958c924f2dc52b03657c5e4dfe78ce3bceb3  manifest.json
```

生成链路：`pilot.py` / `contrast.py` / `query_contrast.py` / `relations.py` → `compile_pool.py` → `translate_atoms.py` → `assemble.py` → 导出。
导出步骤脚本未留存，仅 `manifest.json` 记录来源与 seed（见第 5 节）。

## 2. 模型

训练服务器 8×A100-80GB，仓库 `/mnt/data/cpfs/sprite/workspace/plan_sft/LlamaFactory`。

| 模型 | 配置 | LoRA | lr / epoch / 每卡 batch | train_loss | eval_loss | 训练时长 | 产物 |
|---|---|---|---|---:|---:|---:|---|
| gemma-3-270m-it | `llamafactory_runs/gemma3-270m-lora-sft_261007_v340/` | r32 α64 all | 1e-4 / 2 / 4 | 0.0771 | 0.0067 | 52 分钟 | `saves/gemma3-270m/lora/sft_v340{,_merged}` |
| Qwen3-0.6B | `llamafactory_runs/qwen3-0.6b-lora-sft_261007_v340/` | r32 α64 all | 2e-4 / 2 / 16 | 0.0352 | 0.0040 | 2 小时 26 分钟 | `saves/qwen3-0.6b/lora/sft_v340{,_merged}` |

## 3. 评测基线

评测集（均已冻结在 `planner_val/challenges/`）：`challenge_50.json`、`challenge_v2.json`（241）、`challenge_v3.json`（105）、`probe_v1.json`（107）。

宽松分 = `benchmark_compatible_pass`；严格分 = 另比 step/degree 精确、city 子串、query_type（`planner_val/analysis/v340_generalization.py`，输出 `v340_generalization.json`）。
锚点 = qwen-flash 线上提示词输出，离线去经纬度并展开 repeat 后重评。

| 评测集 | 270M 宽松/严格 | 0.6B 宽松/严格 | 锚点 宽松/严格 |
|---|---:|---:|---:|
| challenge_50（50） | 30 / 30 | 39 / 38 | 43 / 42 |
| challenge_v2（241） | 209 / 199 | 232 / 224 | 195 / 188 |
| challenge_v3（105） | 86 / 84 | 90 / 90 | 82 / 80 |
| probe_v1（107） | 88 / 83 | 94 / 89 | 95 / 93 |

结果目录（`planner_val/results/`，原始结果未入库）：

- `gemma3-270m-lora-v340-{challenge_50,challenge_v2}-20261008-100649`、`…-challenge_v3-20261008-103843`、`…-probe_v1-20261008-144750`
- `qwen3-0.6b-lora-v340-{challenge_50,challenge_v2}-20261008-101101`、`…-challenge_v3-20261008-104104`、`…-probe_v1-20261008-144523`
- `qwen-flash-challenge_50-20261008-150408`、`qwen-flash-challenge_v2-20261008-101414`、`qwen-flash-challenge_v3-20261008-103632`、`qwen-flash-probe_v1-20261008-145956`

四集失败合并明细：`v340_fail_dump.txt`。报告：`v3_summary_zh.md`（技术）、`v3_summary_leadership_zh.md`（汇报）、`probe_v1_review.md`。

## 4. 能力边界（0.6B 严格分失败 62 题）

约 15 题非能力问题：英文默认城市标签口径 8、坐标 schema 遗留 4、旧舞名别名 2、灰题 1。其余约 47 题：

| 类别 | 题数 | 训练覆盖为 0 的形式 |
|---|---:|---|
| 结构算子 | 12 | 各、除了、三连、编号列表、最后…先、边X边Y |
| 过度执行 / 字面触发 | 10 | 点名他人 + 机器人动作词、非愿望式条件句 |
| 词表 / 别名 | 9 | 对掌、顺/逆时针、跟这位 |
| 参数外推 | 7 | 中文“X百”、N 圈；深圳占天气任务 49% |
| 显式通道路由 | 7 | 上网 + 品牌/公司主题 |
| 长句漏项 | 2 | — |

## 5. 已知问题（v340 数据）

- **撤回/改口被拼装改写抹掉**：L2+ 中 retract 线索仅保留 49%、correct 75%；约 1,760 处线索与对象一起被删（算子训练量减半），约 80 处删了线索却保留被撤回内容（标签错误）；writer 还会改数值/方向。judge 只校验 quote 存在与顺序，无法发现。
- **重复词噪声被清洗**：重复保留率 65%；语气词、错别字、无标点基本保留（双字片段保留 95–98%）。
- **英文默认城市标中文“深圳”**（6,921 次），与按语言出标签的口径不一致。
- **导出脚本缺失**：v340 导出不可由代码直接复现。
- `/tmp/v3_cover.py`（覆盖分析脚本）已丢失，结论保留在 `v3_summary_zh.md`。

## 6. 已确认的 v350 口径

- **缺工具回退链**（取代 10-06“缺工具不回退”）：泛查信息 rag_query → 无 rag 走 web_search → 都无则 reply；用户已指定通道（上网查、不联网查、天气、电量、动作/舞蹈）而对应工具缺失 → reply。
- **instruction 一律为输入原文 span**（reply/action/query 均适用）；reply 不设原因枚举，planner 只决定是否 reply、对哪个子任务 reply。
- **城市标签跟随 query 语言**（英文句 → Shenzhen）；严格评分需做城市别名归一。
- 值掩码（虚构/长尾城市、随机数字与多种数字写法）+ 启用工具名/参数名掩码与工具顺序打乱。
- 新增结构算子与最小对（如“小明你坐下”→reply vs “你坐下”→sit_down）。
- L2+ 拼装改为原子逐字保留，只允许连接词与标点衔接。
- 自迭代：能力轴 × 说法风格网格，每格 30–50 道探针，标签由算子计算；失败题不进训练；增量 LoRA（约 1k 新数据 + 3–5k 回放，1 epoch），每 3–5 轮从 base 全量重训；keep/discard 回归门控 + McNemar。
