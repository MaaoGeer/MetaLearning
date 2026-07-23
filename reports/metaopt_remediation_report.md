# MetaOpt remediation report

日期：2026-07-23  
代码基线：`main@d7feeb3781addd6d680336dbbc2ac01b5023eb8d`（修改前）  
范围：P0 功能验证、训练目标、validation/test 协议、逐步指标和产物管理；未运行正式训练或正式实验矩阵。

## 1. 结论

P0 没有发现会使当前 MetaOpt 数值结果失效的更新实现错误：

- DummyMetaOptimizer 与 SGD 在相同 `theta0`、task、学习率、`head_only` scope 和 20 个 step 下逐 tensor、update、logit、probability、loss 及全部分类指标的最大绝对误差为 `0`，成功门槛为 `<1e-6`。
- 3-step、二阶 outer backward 中，所有 LSTM 层、输出层和可学习步长参数的梯度均为 finite 且非零。最小 norm 为 `2.1517180e-6`，最大 norm 为 `1.7152821e-1`。
- 本轮 ddos/seed42/horizon20 的 `meta_artifacts.pt` 与 `best.pt` 中 MetaOpt tensor state 完全一致，`max_abs_diff=0`，tensor hash 均为 `ba17e1f...a0`。`head_only` 实际为 `classifier.weight` 和 `classifier.bias`，共 2 个 tensor、66 个参数。
- step 0 speed 已修复，并保留 `speeds_deprecated_excluding_step0` 兼容旧口径。

因此，审计中“固定 20-step 终点目标促成慢启动策略”仍是当前首要、但尚待 E1/E2 验证的算法假设。P0 证据排除了 Dummy/SGD 方向、基本 flatten/update 路径、二阶元梯度和已审计 checkpoint/scope 不一致这几类首要功能故障；它没有证明 MetaOpt 已优于基线。

已确认的协议/产物问题已在后续运行路径中修复：

1. final-only 仍作为向后兼容默认值，但 multi-step、随机 horizon、SGD anchor + residual/gate/trust region 和 mixed-shot 均成为同一训练入口的配置开关。
2. validation 选择与 test 评估可强制分成两个阶段。`--phase test` 必须读取冻结的 selection receipt，并校验 `theta0`、validation manifest 和 test manifest 的 hash。
3. test-trajectory oracle 仅保留在 `descriptive_only_test_oracle`，schema 明确 `test_selection_allowed=false`。
4. manifest v2 持久化 task、split、attack、shot、seed、support/query window ID、raw-row ID、内容 hash、capture/time block 和窗口 tensor hash；同时报告窗口复用率及保守的 raw-disjoint task 数。
5. 逐 task、逐 step 保存 logits/labels 和分类/校准指标；update/gradient/参数漂移在分片 CSV 中保存。训练及矩阵运行的未来产物按 run/config 隔离，并保存 provenance。

## 2. P0 测试证据

### 2.1 Dummy 与 SGD

测试：`tests/test_metaopt_remediation.py:80`

比较内容：

- 20 步中每个参数 tensor 和 update tensor；
- query logits、probability、support/query loss；
- Accuracy、Precision、Recall、Macro-F1、ROC-AUC、PR-AUC、Brier、ECE。

实测：`dummy_sgd_max_abs_diff=0`。CPU/dtype 均为 float32，因此无需放宽 `1e-6` 容差。

### 2.2 二阶元梯度

测试：`tests/test_metaopt_remediation.py:132`

实测 gradient norm：

| 参数 | norm |
|---|---:|
| `cells.0.weight_ih` | 2.2106378e-5 |
| `cells.0.weight_hh` | 2.1517180e-6 |
| `cells.0.bias_ih` / `bias_hh` | 1.8018996e-5 |
| `cells.1.weight_ih` | 5.0853851e-5 |
| `cells.1.weight_hh` | 2.5291498e-5 |
| `cells.1.bias_ih` / `bias_hh` | 1.8964621e-4 |
| `output.weight` | 4.9260367e-2 |
| `output.bias` | 1.7152821e-1 |
| `raw_lr` | 4.7056579e-5 |

log-sign preprocessing本身无可学习参数，`cells.0.weight_ih` 是其第一个可学习消费者。所有列出的参数均 `grad is not None`、finite、norm > 0；测试使用 `first_order=False`。

### 2.3 checkpoint 与 scope

验证器：`scripts/verify_metaopt_checkpoint.py`  
回执：`reports/p0_checkpoint_scope_receipt.json`

| 检查 | 结果 |
|---|---|
| artifact vs best tensor max diff | 0.0 |
| artifact vs last tensor max diff | 0.0 |
| artifact/best/last tensor hash | `ba17e1f974cd...e7fd1a0` |
| meta initialization hash | `6d165a06464d...2981cb4c` |
| effective/artifact scope | `head_only` / `head_only` |
| 参数名 | `classifier.weight`, `classifier.bias` |
| tensor / 参数量 | 2 / 66 |

本个 checkpoint 的 `best` 与 `last` 文件 hash 不同，但 MetaOpt tensor state 相同；这不是加载错误，而是两个容器保存了相同模型状态及不同的附加内容。

### 2.4 step 0

实现：`src/evaluation/adaptation_speed.py:35-75`  
回归测试：`tests/test_metaopt_remediation.py:157`

新字段的 step 语义为 trajectory index 等于真实 adaptation step。step 0 已达阈值时速度为 0；旧的“忽略 step 0”值仅保存在明确标为 deprecated 的字段中。

## 3. 训练目标和 MetaOpt 更新实现

### 3.1 多步 query loss

实现：`src/meta_learning/outer_loop.py:85-170`

支持：

- `mode: final_only | multi_step`；
- 监督 step，例如 `[1,2,5,10,20]`；
- `uniform`、`early_heavy` 和 `custom` 权重；
- 权重归一化；
- 每个监督 step 的 raw query loss 和 weighted contribution 日志/TensorBoard；
- sampled horizon 可自动加入监督集合。

`final_only` 仍走旧的终点 loss 路径，默认行为不变。推荐 E1 使用 early-heavy，但它不是硬编码唯一选择。

### 3.2 随机 horizon

实现：`src/meta_learning/outer_loop.py:120-150`、`src/trainer/meta_trainer.py:99-155`

- 范围和开关由配置控制，独立 RNG 受 experiment seed 控制；
- 和 multi-step loss 共存；
- validation 固定报告 1/2/5/10/20 step；
- 开启随机 horizon 而 selection 仍为 `final_f1` 时，代码警告并切换至 `curve_auc`，防止又回到仅终点选择。

### 3.3 SGD anchor + learned residual/gate

实现：`src/meta_optimizer/lstm_optimizer.py:54-274`

\[
\Delta\theta=-\alpha g+\operatorname{gate}\,r_\phi
\]

- `learned_delta` 是旧 checkpoint 兼容默认；
- `sgd_residual` 才注册 anchor/gate 参数，因此旧 `state_dict` schema 不变；
- residual output 可零初始化，gate 使用 sigmoid 有界参数化；
- anchor LR 可固定或学习；
- trust region 相对 anchor norm 约束总更新；
- residual 关闭时绕过 residual、trust 和全局 clip，严格退化为 SGD；
- update trace 保存 anchor/residual/total norm、gate、trust scale、clip 和相对 `-alpha*g` 的方向信息。

测试：`tests/test_metaopt_remediation.py:226`。零初始化时初始行为接近 SGD；关闭 residual 时逐 tensor 精确等于 SGD。

### 3.4 mixed-shot

实现：`src/data/task_sampler.py` 的 `MixedShotTaskSampler` 和 `src/trainer/meta_trainer.py:69-80`。默认关闭；E0–E3 仍固定 5-shot。仅当 E1–E3 的目标/结构验证达到门槛后才运行 E4。

## 4. validation/test 协议

### 4.1 固定 task manifest

实现：`src/evaluation/task_manifest.py`、`scripts/generate_eval_task_manifest.py`

manifest v2 包含：

- `task_id`、split、attack、shot、task seed；
- support/query local window ID；
- raw-row ID、raw-row 内容 hash、window tensor hash；
- segment/capture/time block、order start/end；
- base initialization hash 和 manifest sidecar SHA-256。

`manifest_reuse_statistics` 输出 task 数、unique task hash、window occurrence/unique/reuse rate，以及保守的 greedy raw-disjoint task count。若 raw-disjoint 数小于 sampled task 数，评估器会警告并将 `independent_replication_claim_allowed=false`；这避免把伪重复直接表述为独立重复，但该 greedy 数不是严格统计有效样本量。

validation/test manifest 的 raw-row 集合有任何重叠会直接报错。相同 seed 下不同方法复用同一 manifest；不同 MetaOpt 配置只要 `theta0` hash 相同，也能复用同一 manifest。

### 4.2 冻结选择回执

实现：`scripts/run_experiments.py:84-96,598-630,798-849,1100-1158`

推荐流程：

1. `--phase validation`：只在 validation manifest 上选 SGD/Adam LR 和每个方法的 stop step，生成 `validation_selection.json`，其中 `test_metrics_used=false`。
2. 审核 E0–E4 的 validation 结果后预先确定配置。
3. 仅对被选中的配置运行一次 `--phase test --selection-receipt ...`。

test 阶段不会重新进行 LR 网格搜索。若 θ0、validation manifest hash 或 test manifest hash 与回执不同，运行被拒绝。`--phase both` 只为旧脚本兼容保留，并打印协议警告，不应用于论文实验。

### 4.3 指标和落盘

未来评估会生成：

- `adaptation_curves.csv`：每 task/step 的 Accuracy、Precision、Recall、Macro-F1、ROC-AUC、PR-AUC、attack recall、FPR、Brier、ECE；
- `prediction_trajectories.npz` + schema：逐 task/step logits 和 labels；
- `step_diagnostics.csv`：support loss、预测分布、校准指标、`||theta_t-theta_0||`；
- `update_analysis.csv`：gradient/update norm、update/gradient ratio、anchor/residual/gate/trust/clip 和方向信息；
- `results.json` + `result_schema.json`：主结果和 descriptive-only oracle 的边界；
- `provenance.json`：git commit、effective config、cache key、raw file size/mtime/hash、manifest/checkpoint/artifact hash。

训练曲线、TensorBoard、effective config、checkpoint 和图像写入该 run 的 artifact directory。矩阵 launcher 进一步加入 `attack/fraction/seed/horizon/config_id`，不会覆盖旧结果。

### 4.4 基线公平性

Adam 默认网格已扩为 `[0.001,0.003,0.01,0.03,0.1,0.3]`。SGD/Adam 均只在 validation manifest 上选 LR。若最优点在上下边界，结果记录 `fully_tuned_claim_allowed=false` 并输出警告。

## 5. 配置项和默认值

| 配置 | 默认值 | 说明 |
|---|---|---|
| `meta_optimizer.update_mode` | `learned_delta` | 旧实现；E3/E4 用 `sgd_residual` |
| `meta_optimizer.anchor_lr` | `0.1` | SGD anchor |
| `meta_optimizer.learnable_anchor_lr` | `false` | 是否学习 anchor LR |
| `meta_optimizer.residual_enabled` | `true` | residual 开关 |
| `meta_optimizer.residual_zero_init` | `true` | residual 输出零初始化 |
| `meta_optimizer.gate_init` | `0.01` | 初始 gate 概率 |
| `meta_optimizer.learnable_gate` | `true` | 是否学习 gate |
| `meta_optimizer.trust_region_factor` | `null` | 相对 anchor norm 的约束；E3 候选为 2.0 |
| `meta.query_objective.mode` | `final_only` | 旧行为 |
| `meta.query_objective.supervised_steps` | `[20]` | multi-step 监督点 |
| `meta.query_objective.weighting` | `uniform` | `uniform/early_heavy/custom` |
| `meta.query_objective.custom_weights` | `[]` | 自定义未归一化权重 |
| `meta.query_objective.early_heavy_power` | `0.5` | \(w_t\propto t^{-p}\) |
| `meta.random_horizon.enabled` | `false` | 随机 horizon 默认关闭 |
| `meta.random_horizon.min_steps/max_steps` | `1/20` | 采样范围 |
| `meta.mixed_shot.enabled` | `false` | 按要求暂缓默认启用 |
| `meta.mixed_shot.shots` | `[1,3,5,10]` | 候选 shot |
| `train.validation.checkpoints` | `[1,2,5,10,20]` | 固定验证点 |
| `train.validation.selection_metric` | `final_f1` | E1–E4 改为 `curve_auc` |
| `provenance.hash_raw_data` | `true` | 未来 provenance 记录原始文件 SHA-256 |

注意：`configs/base.yaml` 的 `meta.inner_steps` 仍为历史默认 5；E0–E5 命令显式覆盖为 20。旧 artifact 的 effective config 优先于文件名。

## 6. 已执行验证

| 验证 | 命令/范围 | 结果 |
|---|---|---|
| Python 静态编译 | `python -m compileall -q src scripts train_meta.py` | PASS |
| 全量 pytest | `python -m pytest -q` | `55 passed, 1 warning in 7.58s` |
| P0/新功能定向测试 | `python -m pytest tests/test_metaopt_remediation.py -q -s` | `8 passed in 3.21s` |
| checkpoint/scope | `scripts/verify_metaopt_checkpoint.py` | PASS，见回执 |
| 旧评估兼容 smoke | 1 shot、1 val task、1 test task、1 step、CPU | PASS，约 50.5 秒；首次生成新 cache |
| validation-only smoke | 同一固定 manifests，CPU | PASS，12.7 秒 |
| frozen test-only smoke | 读取 selection receipt，CPU | PASS，8.6 秒；未重新网格搜索 |
| schema/产物检查 | NPZ/CSV/JSON 手工断言 | PASS；logits shape `(3,2,20,2)` |
| diff whitespace | `git diff --check` | PASS |

pytest 唯一 warning 来自已有 significance test 的近相同数值造成 SciPy precision-loss；测试通过，与新训练/协议逻辑无关。

smoke 证据：

- `reports/smoke_eval_protocol_validation/validation_selection.json`
- `reports/smoke_eval_protocol_validation/result_schema.json`
- `reports/smoke_eval_protocol_test/result_schema.json`
- `reports/smoke_eval_protocol_test/prediction_trajectories.npz.schema.json`
- `reports/p0_checkpoint_scope_receipt.json`

## 7. 未运行的验证和实验

未运行 E0–E5 的 10-epoch GPU 元训练、2 attacks × 2 seeds 评估或正式 6 attacks × 5 seeds × 4 shots 矩阵，因为本任务明确禁止耗时训练和正式矩阵。因而目前：

- 不能声称 multi-step 能把 Step 1 提高 0.08；
- 不能声称 random horizon 或 residual/gate 优于 final-only；
- 不能断定审计中的慢启动因果链已被验证；
- 不能以 smoke 的 1-task 指标作性能结论；
- 不能证明 Adam 的扩展网格不再命中边界；
- 不能证明 botnet 的 task 独立性足够。

## 8. 最小 E0–E5 实验矩阵

固定：ddos/botnet，seed 42/62，5-shot，horizon 20，10 epochs（正式 30 epochs 的约 1/3），相同 validation/test manifest。

| ID | 改动 | 用途 |
|---|---|---|
| E0 | `final_only + learned_delta + fixed horizon` | 当前基线 |
| E1 | multi-step `[1,2,5,10,20]` + early-heavy | 验证终点目标是否为首因 |
| E2 | E1 + random horizon 1–20 | 验证 horizon 鲁棒性 |
| E3 | E1 + SGD anchor/residual/gate/trust，固定 horizon | 隔离 learned-delta 结构 |
| E4 | E3 + mixed-shot | 仅 E1–E3 通过后运行 |
| E5 | E0 artifact + 扩展 Adam validation LR grid | 无需元训练 |

成功门槛保持预注册值：

1. P0 全部通过（已满足）；
2. MetaOpt Step 1 相对当前版本至少 `+0.08`；
3. MetaOpt−SGD Step 1 不低于 `-0.05`；
4. MetaOpt−SGD Curve AUC 不低于 `-0.02`；
5. Final Macro-F1 相对当前 MetaOpt 下降不超过 `0.01`；
6. ddos/botnet、seed 42/62 改进方向一致；
7. 无 NaN/Inf，clip ratio `<5%`；
8. median update/grad 随 step 不跨越两个数量级；
9. 所有选择仅用 validation，test 只运行一次。

解释规则：E1 成功支持“训练目标首要”；E1 失败但 E3 成功支持“纯 learned-delta 结构首要”；ddos 成功而 botnet 失败时优先审计独立窗口/task；任何新 P0 失败都停止性能解释。

## 9. 可在服务器执行的 PowerShell 命令

以下命令只定义和启动明确的最小矩阵。建议先单跑 E0/ddos/42 校准时间和显存，再展开循环。E4 不在首轮数组中。

```powershell
$Known = @{
  ddos   = '["botnet","bruteforce","dos","heartbleed","infiltration","portscan","webattack"]'
  botnet = '["bruteforce","ddos","dos","heartbleed","infiltration","portscan","webattack"]'
}
$Variants = @(
  @{ Id="E0_final_only"; Extra=@(
      "meta.query_objective.mode=final_only",
      "meta.random_horizon.enabled=false",
      "meta_optimizer.update_mode=learned_delta"
  )},
  @{ Id="E1_multistep"; Extra=@(
      "meta.query_objective.mode=multi_step",
      "meta.query_objective.supervised_steps=[1,2,5,10,20]",
      "meta.query_objective.weighting=early_heavy",
      "meta.random_horizon.enabled=false",
      "meta_optimizer.update_mode=learned_delta",
      "train.validation.selection_metric=curve_auc"
  )},
  @{ Id="E2_multistep_random_horizon"; Extra=@(
      "meta.query_objective.mode=multi_step",
      "meta.query_objective.supervised_steps=[1,2,5,10,20]",
      "meta.query_objective.weighting=early_heavy",
      "meta.random_horizon.enabled=true",
      "meta.random_horizon.min_steps=1",
      "meta.random_horizon.max_steps=20",
      "meta_optimizer.update_mode=learned_delta",
      "train.validation.selection_metric=curve_auc"
  )},
  @{ Id="E3_sgd_residual"; Extra=@(
      "meta.query_objective.mode=multi_step",
      "meta.query_objective.supervised_steps=[1,2,5,10,20]",
      "meta.query_objective.weighting=early_heavy",
      "meta.random_horizon.enabled=false",
      "meta_optimizer.update_mode=sgd_residual",
      "meta_optimizer.anchor_lr=0.1",
      "meta_optimizer.residual_zero_init=true",
      "meta_optimizer.gate_init=0.01",
      "meta_optimizer.trust_region_factor=2.0",
      "train.validation.selection_metric=curve_auc"
  )}
)

foreach ($variant in $Variants) {
  foreach ($attack in @("ddos","botnet")) {
    foreach ($seed in @(42,62)) {
      $run = "outputs/metaopt_remediation/$($variant.Id)/$attack/seed_$seed/horizon_20"
      $args = @(
        "train_meta.py",
        "--config", "configs/base.yaml",
        "--dataset", "configs/datasets/cicids2017.yaml",
        "--out", "$run/meta_artifacts.pt",
        "--override",
        "experiment.seed=$seed",
        "data.unknown_class=$attack",
        "data.known_classes=$($Known[$attack])",
        "data.k_shot=5",
        "meta.inner_steps=20",
        "train.meta_epochs=10",
        "train.early_stopping.enabled=true",
        "train.early_stopping.patience=4",
        "train.validation.checkpoints=[1,2,5,10,20]"
      )
      $args += $variant.Extra
      python @args
      if ($LASTEXITCODE -ne 0) { throw "training failed: $($variant.Id)/$attack/$seed" }
    }
  }
}
```

生成每个 attack/seed 唯一的一对 manifest。它们以 E0 的 `theta0` 为锚，E1–E4 只有在相同 seed 产生相同 `theta0` hash 时才能复用：

```powershell
foreach ($attack in @("ddos","botnet")) {
  foreach ($seed in @(42,62)) {
    $artifact = "outputs/metaopt_remediation/E0_final_only/$attack/seed_$seed/horizon_20/meta_artifacts.pt"
    $manifestDir = "outputs/metaopt_remediation/manifests/$attack/seed_$seed"
    python scripts/generate_eval_task_manifest.py --artifacts $artifact --out "$manifestDir/validation_5shot.json" --shot 5 --tasks 30 --task-seed ($seed + 1) --split val
    python scripts/generate_eval_task_manifest.py --artifacts $artifact --out "$manifestDir/test_5shot.json" --shot 5 --tasks 100 --task-seed ($seed + 1001) --split test
    if ($LASTEXITCODE -ne 0) { throw "manifest generation failed: $attack/$seed" }
  }
}
```

仅运行 validation。E5 使用 E0 artifact 和已扩展的 Adam grid，不产生额外元训练：

```powershell
$EvalVariants = @("E0_final_only","E1_multistep","E2_multistep_random_horizon","E3_sgd_residual")
foreach ($variant in $EvalVariants) {
  foreach ($attack in @("ddos","botnet")) {
    foreach ($seed in @(42,62)) {
      $root = "outputs/metaopt_remediation/$variant/$attack/seed_$seed/horizon_20"
      $m = "outputs/metaopt_remediation/manifests/$attack/seed_$seed"
      python scripts/run_experiments.py --artifacts "$root/meta_artifacts.pt" --out "$root/validation" --phase validation --validation-task-manifest "$m/validation_5shot.json" --test-task-manifest "$m/test_5shot.json" --override "compare.shots=[5]" "adaptation_speed.max_steps=20" "adaptation_speed.checkpoints=[0,1,2,5,10,20]"
      if ($LASTEXITCODE -ne 0) { throw "validation failed: $variant/$attack/$seed" }
    }
  }
}
```

审核 validation 后，把 `$ChosenVariant` 固定为一个配置。下列 test 命令每个 attack/seed 只运行一次：

```powershell
$ChosenVariant = "E1_multistep"  # 只能根据 validation 结果预先填写
foreach ($attack in @("ddos","botnet")) {
  foreach ($seed in @(42,62)) {
    $root = "outputs/metaopt_remediation/$ChosenVariant/$attack/seed_$seed/horizon_20"
    $m = "outputs/metaopt_remediation/manifests/$attack/seed_$seed"
    python scripts/run_experiments.py --artifacts "$root/meta_artifacts.pt" --out "$root/test_once" --phase test --selection-receipt "$root/validation/validation_selection.json" --validation-task-manifest "$m/validation_5shot.json" --test-task-manifest "$m/test_5shot.json" --override "compare.shots=[5]" "adaptation_speed.max_steps=20" "adaptation_speed.checkpoints=[0,1,2,5,10,20]"
    if ($LASTEXITCODE -ne 0) { throw "test failed: $ChosenVariant/$attack/$seed" }
  }
}
```

E4 仅在 E1–E3 达到门槛后，将 E3 训练命令增加：

```powershell
"meta.mixed_shot.enabled=true"
"meta.mixed_shot.shots=[1,3,5,10]"
```

并使用输出 ID `E4_sgd_residual_mixed_shot`。不要先看 test 再决定是否运行 E4。

## 10. 成本估计

| 操作 | 单 run 预计时间 | 显存 | 输出 |
|---|---:|---:|---|
| P0/pytest | CPU < 15 秒 | 0 | 控制台/临时 pytest 目录 |
| manifest 30+100 tasks（warm cache） | CPU 10–60 秒 | 0 | `outputs/metaopt_remediation/manifests/...` |
| 10-epoch、20-step head-only 二阶训练 | 单现代 GPU 20–90 分钟 | 建议 4 GB，保守预留 8 GB | variant/attack/seed/horizon |
| validation：30 tasks，SGD 4 点 + Adam 6 点 + 3 methods | 单 GPU 5–30 分钟 | 通常 <2 GB，建议预留 4 GB | `.../validation` |
| 一次性 test：100 tasks × 3 methods | 单 GPU 5–30 分钟 | 通常 <2 GB，建议预留 4 GB | `.../test_once` |
| E0–E3 全部 16 个训练（串行） | 约 5–24 GPU 小时 | 同上 | `outputs/metaopt_remediation` |

时间是基于当前小模型、head-only 和本机 smoke 的工程估计，不是实测服务器 SLA。先跑 E0/ddos/42；若显存或时间超过范围，缩小 `tasks_per_epoch` 而不是改 test 或成功门槛，并将实际预算写入 effective config。

## 11. 兼容性和风险

- 旧配置缺少新字段时使用旧 `learned_delta/final_only/fixed-shot/fixed-horizon` 默认。
- 旧 learned-delta checkpoint 不注册 residual 参数，已通过现有 artifact 的 end-to-end smoke 加载。
- manifest reader兼容 schema v1；v1 仍绑定整个 artifact hash，不能跨配置复用。新消融必须用 v2。
- `--phase both` 兼容旧调用，但不符合新的论文协议。
- 新 provenance 不能追溯性地修复旧 cache；旧结果仍应保留审计报告中的 cache 风险说明。
- raw-disjoint greedy count 只是一项保守诊断，不等价于正式有效样本量估计。
- residual/gate/trust 的功能测试通过，但其泛化收益、clip ratio 和 update/grad 动力学尚无训练证据。
- E3 的 `anchor_lr=0.1`、`trust_region_factor=2.0` 是预注册候选，不应根据 test 调整；若需搜索，只能建立独立 validation-only 小网格。

## 12. 放行判断

运行 E0–E3 最小矩阵：**是**。P0、配置路径、固定 manifest、validation-only、frozen test-only、逐步产物和旧 checkpoint 兼容 smoke 均通过。

运行 E4：**否，暂不放行**。必须先满足 E1–E3 的数值门槛和一致性条件。

运行正式 6 attacks × 5 seeds × 4 shots 矩阵：**否**。尚缺 E0–E3 的验证结果、Adam 是否仍命中边界、botnet 独立 task 统计、clip/update 动力学和一次性 test 协议的完整小矩阵证据。

## 13. 服务器预运行追加发现：stride-overlap sampler

2026-07-23 的首次 E0/ddos/seed42 服务器启动在构造固定 meta-validation
task 时中止，错误为 `support/query 原始样本重叠: [(133, 134)]`。服务器
effective data protocol 使用 `window_size=16, stride=8`，相邻窗口会共享 raw
rows。定位确认旧 `FewShotTaskSampler` 在逐类别采样时只把已选 support 加入
`forbidden`，没有把前一类别的 query 加入下一类别 support 的 forbidden 集合；
因此最终审计正确地拦截了跨类别 raw-row overlap。

修复保持严格 overlap 检查开启：

- 已选 query 同样加入后续类别的 forbidden 集合；
- forbidden 检查和 within-selection internal-overlap 检查解耦；
- 随机贪心选择增加 32 次有界重试，避免一次不利随机顺序造成虚假“窗口不足”；
- 新增 stride-overlap 跨类别回归测试，旧实现可由 seed 1 稳定复现失败。

修复后定向 sampler 测试 `4 passed`，全仓测试更新为
`56 passed, 1 warning`。失败发生在第一个 epoch 前，没有生成可用于分析的
E0 artifact；续跑应保留原输出目录并使用 launcher 的 `-Resume`。
