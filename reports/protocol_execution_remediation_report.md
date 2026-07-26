# MetaOpt clean rerun 执行协议修复报告

## 执行摘要

本次修改仅修复实验执行协议，不改变模型、MetaOpt、训练目标、预注册
E0–E3 定义、数据切分内容、seed、shot、horizon、基线学习率网格或评价门槛。
修改前基准为
`27f4fcb1494c9dd1237767b1b17a5bc61e20ac0a`，实际本地仓库为
`E:\Progect\MetaLearning`。修改前工作区为 clean，证据保存在
`reports/protocol_execution_prechange_receipt.json`。

根因是三个入口共享了一个隐式构造全部 adaptation split 的 pipeline：

1. meta-training 虽只消费 meta-train/meta-val，仍构造 adaptation validation/test；
2. manifest 生成入口先构造完整 pipeline，再从中选 validation 或 test；
3. `run_experiments.py` 在 phase 分支前同时创建 validation/test sampler 和
   manifest；
4. 旧 PowerShell `Prepare` 阶段串联 16 次训练、validation manifest、test
   manifest 和 validation，且向 validation 命令传递 test manifest；
5. 旧 runner 的 Smoke 不是独立单元，完成回执也不足以证明源码、split 访问
   和必要产物完整性。

## 从父提交到修复状态的 diff 摘要

父提交固定为
`27f4fcb1494c9dd1237767b1b17a5bc61e20ac0a`。变更面仅包括：

- pipeline 与既有调用点的显式 dataset role；
- training/manifest/evaluation 的 commit/source-state 门禁和回执；
- validation-only 逐步动力学落盘；
- clean validation 审计与 frozen receipt 生成；
- 7 阶段 PowerShell runner；
- data-free protocol plan；
- CPU sentinel/runner/provenance 测试；
- README 和本报告。

以下路径没有 diff：`configs/`、`src/models/`、`src/meta_optimizer/`、
`src/meta_learning/`、`src/trainer/meta_trainer.py`。因此模型、MetaOpt、训练
目标、E0–E3 配置和超参数未改变。

## Pipeline dataset role

`build_pipeline` 现在强制调用者显式传入 `adaptation_dataset_role`，无默认值：

| Role | meta train/val | adapt validation | adapt test | 用途 |
|---|---:|---:|---:|---|
| `none` | 是 | 否 | 否 | MetaOpt meta-training |
| `validation` | 否 | 是 | 否 | validation manifest / validation evaluation |
| `test` | 否 | 否 | 是 | test manifest / frozen test evaluation |
| `all` | 是 | 是 | 是 | 仅保留旧诊断兼容入口 |

请求未构造的数据集会立即抛出异常，因而 validation 代码无法静默回退到 test，
test 代码也无法静默读取 validation。meta-training 真实使用字段经调用链核对为
feature/window metadata、meta train sampler、meta validation sampler 和类别信息，
不需要 unknown adaptation validation/test，因此使用 `none`。

## Manifest 生成隔离

`generate_eval_task_manifest.py` 要求显式 `--split`：

- `val`/`validation` 绑定 role `validation` 和 `adapt_val_dataset`；
- `test` 绑定 role `test` 和 `adapt_test_dataset`；
- split 缺失或未知由 argparse 立即拒绝；
- 不再构造完整 pipeline 后选择 split；
- clean rerun 的 validation manifest 可以由配置和固定 seed 预先生成，绑定
  tensor-level `theta0` hash，后续 artifact 必须与之匹配；
- 已存在 manifest 或 sidecar 时拒绝覆盖。

生成回执包含 requested/effective split、dataset role、effective dataset、
dataset fingerprint、effective config hash、manifest/sidecar hash、theta0 hash、
artifact hash、源码状态、运行环境、时间、退出状态和 split
constructed/accessed 标志。validation 成功回执明确记录：

```text
effective_dataset=adapt_val_dataset
test_dataset_constructed=false
test_dataset_accessed=false
metaopt_training_ran=false
```

## Validation/test 阶段路由

`run_experiments.py` 在加载 artifact 或构造 pipeline 前先执行 phase 参数门禁：

- `validation` 拒绝 `--task-manifest`、`--test-task-manifest` 和
  `--selection-receipt`，只接受 validation manifest；
- `test` 拒绝 validation manifest，强制要求现有 test manifest 和冻结 selection
  receipt；
- frozen test 不创建随机 test sampler，只按 manifest window ID 重建 task；
- `both` 仅作旧兼容，日志明确 deprecated；论文 runner 不包含该参数；
- validation 逐步保存 prediction、诊断、gradient/update、clip 和 nonfinite
  产物，供门槛审计；
- 成功、失败和中断均写 completion receipt；只有 exit code 为 0 且全部必要产物
  存在时才能标记 success。

test 分支从冻结 receipt 读取 SGD/Adam LR 和每个方法的 stop step，不重新训练、
不执行 validation 搜索、不重新选择 LR/checkpoint/stop step，也不生成 manifest。

## Runner 分阶段

`scripts/run_metaopt_minimal_experiments.ps1` 现在只有以下显式阶段：

1. `Preflight`：commit、clean worktree、source-state 和 tracked source 门禁；
2. `Smoke`：仅 E0/ddos/seed42，1 epoch，validation-only；
3. `PrepareValidation`：只生成 4 个 validation manifest，不展开训练；
4. `RunValidation`：精确展开 4 variants × 2 attacks × 2 seeds = 16 个串行
   validation run；
5. `AuditValidation`：只读取新 clean namespace，核验 16/16、fairness、
   provenance、动力学和预注册 gate；
6. `PrepareFrozenTest`：仅在 validation gate 通过后生成 test manifest 和冻结
   receipt；
7. `FrozenTest`：还需显式 `ALLOW_FROZEN_TEST=YES`，且输出路径必须不存在。

默认不会串联到下一阶段。每个阶段有独立 completion receipt；Smoke 和每个正式
validation run 还有 run-level receipt。已存在成功回执或输出目录时拒绝重复或
覆盖。实验输出位于 `.gitignore` 覆盖的 `outputs/`，不会把合法运行产物误判为
tracked source 污染。

E0–E3 override 只有一个实现来源：
`src/utils/experiment_protocol.py`。旧 runner 中的预注册值原样迁移，没有重新
设定 multi-step 权重、random horizon、gate、anchor LR 或 trust region。

## Provenance 和 completion receipt

门禁同时验证：

- full Git commit；
- `git status --porcelain` 为空；
- 可选的、基于当前 commit Git blob 的 portable source-state hash；
- runner、配置、pipeline、task/manifest、模型构建、MetaOpt、trainer 和指标
  关键文件均由 Git 跟踪并具有 SHA-256。

正式 gating 字段使用算法
`git-commit-blob-path-content-sha256/v1`：对排序后的仓库相对路径建立
`path -> {git_blob_oid, blob_sha256}` 映射，再对 canonical JSON 求 SHA-256。
因此该值不受 CRLF/LF、`core.autocrlf` 或操作系统影响。工作区原始字节另以
`worktree-path-bytes-sha256/v1` 记录，并明确标记为
`diagnostic_only=true, gating=false`；它可用于解释检出差异，但不得用于跨机器
放行。

阶段/run 回执包含适用的：

- commit、parent、clean 状态、source-state 和逐文件 hash；
- effective config hash；
- dataset role/fingerprint；
- manifest/sidecar hash；
- theta0、artifact、checkpoint hash；
- Python/PyTorch/CUDA、hostname 和命令行；
- 开始/结束时间、duration、exit code 和 completion status；
- validation/test constructed/accessed；
- MetaOpt training 是否运行；
- 输出路径和必要产物逐文件 hash。

## Validation 审计规则

`audit_clean_validation.py` 只接受名称含目标 commit 短 hash 的 clean rerun
namespace，只读取其中 16 个正式 validation run。它验证：

- 16/16、无重复 key；
- 每个 run commit 精确、worktree clean、validation-only；
- 同 attack/seed 下 E0–E3 的 theta0、manifest、dataset fingerprint 和 SGD
  Step 1/Curve AUC/Final 一致；
- nonfinite、clip ratio 和 update/gradient ratio span；
- botnet validation manifest 的窗口复用与 raw-disjoint task 数。

候选必须在 4 个 attack/seed 单元全部通过原门槛。若多个候选通过，冻结规则是：
在通过全部门槛的候选中选择 mean validation Curve AUC 最高者。若无候选全部
通过，`selected_variant=null` 且不得进入 frozen test。

## CPU 验证

新增测试使用可区分的 sentinel validation/test dataset 和 monkeypatch，错误
split 的构造或访问会立即失败。覆盖：

- meta-training 不构造 adaptation test；
- validation/test role 只构造请求的数据集；
- role 缺失/未知立即失败；
- validation/test manifest 生成互不构造或访问另一 adaptation dataset；
- validation phase 拒绝所有 test manifest 参数；
- test phase 拒绝 validation manifest，并要求冻结输入；
- Smoke 1 run、PrepareValidation 4 manifests、RunValidation 16 runs；
- 论文 runner 不使用 deprecated `both`；
- dirty worktree 和错误 commit 门禁；
- success/failed/interrupted 回执字段；
- success receipt 的必要产物完整性。

验证命令和结果：

```text
python -m pytest -q tests/test_protocol_execution_isolation.py \
  tests/test_adam_validation_split_safety.py tests/test_experiments.py \
  tests/test_metaopt_remediation.py tests/test_protocol_fix.py
40 passed

python -m pytest -q
83 passed, 1 expected numerical precision warning

python -m compileall -q src scripts tests train_meta.py
PASS

PowerShell parser check
PASS

Runner dry-run job counts:
Preflight=0, Smoke=1, PrepareValidation=4, RunValidation=16,
AuditValidation=1, PrepareFrozenTest=4, FrozenTest=8
No output directory created
```

最终提交前还会重新运行上述验证、`git diff --check` 和禁止修改范围检查。

## 限制与下一步

本轮没有调用 CUDA、没有启动 Smoke、E0–E3 training、Adam validation 或任何
test。当前只具备在新 commit push 并在服务器精确 checkout 后，重新执行静态
Smoke preflight 的条件。GPU Smoke 仍需用户下一轮明确授权。
