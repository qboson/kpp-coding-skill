# Kaiwu-PyTorch-Plugin（KPP）规则

## 安装包与依赖（2026-10-01 实测）

**pip 包名：`kaiwu-torch-plugin`；Python 导入：`kaiwu.torch_plugin`。**
源码仓库名是 `kaiwu-pytorch-plugin`，历史文档的 0.2.0 指源码快照，不是安装命令。
当前实测安装版为 **0.3.0**，其 `Requires-Dist` 明确固定：

| 依赖 | 版本 |
|---|---|
| kaiwu | 1.3.1 |
| torch | 2.7.0 |
| numpy | 2.2.6 |

本次在 macOS arm64、Python 3.10.21 上验证了导入、真实 SDK SA 采样和 RBM 三步训练
（形状/二值检查、有限损失与梯度、参数确实更新）。其他系统应重新检查 wheel 支持，
未来包版本以实际元数据为准。可用 `importlib.metadata.requires('kaiwu-torch-plugin')`
查看当前依赖声明。

`kaiwu==1.3.1` 只提供 cp310 wheel（macOS x86_64/arm64、manylinux x86_64、Windows amd64；
无 Linux aarch64），而 PyPI 上的 KPP 声明 `requires_python>=3.8`。因此在 3.11/3.12 上
dry-run 会报 kaiwu 没有匹配版本。这表示需要独立的 3.10 环境，不是结束条件。

## 自动安装流程

当用户要求使用 KPP，或任务选择了需要 SDK 采样的能量模型路线，主动完成以下步骤。
已经明确的项目环境不重复询问；普通 ML 任务不要仅因出现“训练”“生成”就安装 KPP。
已有本次依赖与导入检查结果时直接复用，进入任务模型的训练与验证。

1. 找到用户指定/项目使用的解释器，检查 Python、KPP、Kaiwu 和 torch 的实际版本。
2. 兼容且已安装时直接复用。缺包时才用目标解释器运行
   `python -m pip install --dry-run kaiwu-torch-plugin`，确认兼容后安装。
3. 如果解释器不是 3.10、计划会降级项目原有 Kaiwu 1.4.x 或破坏其他依赖，且任务允许隔离环境，
   在允许修改的项目目录内创建独立 Python 3.10 环境：

   ```bash
   python3.10 -m venv .venv-kpp
   ./.venv-kpp/bin/python -m pip install kaiwu-torch-plugin
   ./.venv-kpp/bin/python -m pip check
   ```

   Windows 对应 `.venv-kpp\Scripts\python.exe`。已有 uv 可用
   `uv venv --python 3.10 --seed .venv-kpp` 创建环境。后续命令显式指定该环境的解释器；
   绝对路径或项目内相对路径均可，按工具允许的调用方式执行，
   不依赖终端中 `pip` 恰好属于哪个环境。不强行把当前 KPP 0.3.0 装到 SDK 1.4.x 中，
   不用 `--no-deps` 绕开依赖冲突。
4. 从技能目录运行随包提供的验证脚本（把 `python` 换为 KPP 环境的解释器）：

   ```bash
   python scripts/verify_kpp.py --imports-only
   # 输出 KPP_IMPORT_OK：版本可读取且六个核心组件与 SA 导入成功；尚未验证求解授权。
   python scripts/verify_kpp.py
   # 输出 KPP_TRAIN_OK：真实 SA 采样、RBM 训练和参数更新通过。
   # 输出 KPP_LICENSE_MISSING 并以退出码 2 结束：未找到 license，设置环境变量后重试。
   ```

同一环境的依赖检查和导入检查各执行一次；验证脚本已包含解释器与版本，直接复用
输出保存记录。原环境前后比较只覆盖任务明确要求的项目，不默认扫描全部文件。
安装任务不额外构造模型、探测内部属性或试跑训练；已有检查结果不重复执行。
使用用户提供的离线包时，安装计划和实际安装都带 `--no-index --find-links <离线目录>`；
wheel 元数据直接从压缩包读取，无需解包后注入环境。

默认训练检查使用 4 个可见单元 + 3 个隐藏单元，在 CPU 上运行 3 步；不调用云端 CIM。
任务已有可运行程序时，直接验证该程序，无需再执行额外的通用训练检查。
只验证安装时用 `--imports-only`。任务提供兼容本地采样器时直接复用，授权需求由采样器决定；
选择 SDK SA/CIM 或运行验证脚本默认训练检查时，使用用户已有的授权配置。
需要初始化时，从用户提供的本地环境变量读取 `KAIWU_USER_ID` / `KAIWU_SDK_CODE`，
在该解释器里调用 `kaiwu.license.init(user_id=..., sdk_code=...)`；它写入 license 文件，
每个环境一次即可。验证脚本默认模式在两个变量都存在时自动执行这一步。
不要调用 `kaiwu.license.ensure_license()`：无 license 文件时它会在控制台提示输入
user_id/sdk_code，SA `solve()` 也会提示 `Please enter user ID:`；非交互运行时会卡住或报
`ValueError: EOF when reading a line`。具体账号值不写入技能、示例或 Git。

如果没有兼容 wheel、依赖解析失败、或用户要求必须共用一个与 KPP 冲突的固定环境，
从版本、wheel 元数据及安装计划确定冲突后，保存记录并结束；不解包注入 `PYTHONPATH`
或用 `--target` 绕开依赖继续导入。检查日志与临时文件同样只能写入任务允许的目录；
保存 pip 输出时用项目内如 `./环境检查.log`，不要重定向到 `/tmp` 或其他目录。
不要跳过依赖检查或循环重装。所选 SDK 求解器授权缺失时仍可完成安装与导入检查，
明确标注该求解器未验证；任务必须使用它时再索取缺少的授权信息。KPP/RBM 检查通过不代表特定 QVAE、
QDiffusion 架构或云端 CIM 已验证，后续按实际任务测试。

## 范围与公共模型

从 `kaiwu.torch_plugin` 导入模型：

- `RestrictedBoltzmannMachine`：二部可见/隐藏能量模型。
- `BoltzmannMachine`：全连接能量模型。
- `UnsupervisedDBN`：RBM 堆栈。
- `QVAE`：构造时注入 encoder/decoder/bm/sampler，或子类实现四个 `_create_*` 工厂。
- `QDiffusion`、`QDiffusionConfig`、`EnergyModel`：引导式离散生成。
- `FeatureSelectionWrapper`、`QuadraticLinearSolver`：MAIFS 特征选择，为 PyTorch 模型
  维护 0/1 输入掩码，周期性用梯度+Hessian 构造 QUBO 求解。

KPP 是 PyTorch 包。不依附模型的常规组合优化问题直接用 Kaiwu SDK 的 `QuboModel`，不要
绕道 RBM/BM 或特征选择包装器。

## 采样器契约与 BM/RBM 训练

`AbstractBoltzmannMachine.sample(sampler)` 把 NumPy Ising 矩阵传给
`sampler.solve`。采样器必须返回形状 `(sample_count, num_nodes + 1)`、取值
`-1/+1` 的数组；最后一列是参考自旋。KPP 把该输出转成 `0/1` 的
`torch.FloatTensor` 并移到 `model.device`。

BM 的正相直接使用观测变量；RBM 的正相使用 `get_hidden(batch, bernoulli=True)`
补出隐藏状态。按任务需要的结构选择，训练函数见 [BM/RBM 对照示例](examples.md)。

`batch`、模型及手动创建的张量保持在 `model.device` 上。
`get_hidden`/`get_visible` 处理二元 0/1 态；`sample` 尽管接收 SDK 的自旋
样本，返回的已是二元态。调用 `objective` 前先核对正负相宽度一致：
BM 为观测节点数，RBM 为 `num_visible + num_hidden`。

条件 BM 采样用 `BoltzmannMachine.condition_sample(sampler, s_visible)`。
它对每个可见行调用一次求解器，返回拼接的可见+隐藏二元态。
固定的总是前 `s_visible.shape[1]` 个节点；已知位置逐行不同时，先按“已知在前”置换
`linear_bias` 与对称化 `quadratic_coef` 建临时模型，采样后再逆置换列。耗时随条件行数线性增长。需要本地
Gibbs 采样而非 SDK 求解器时，用 `gibbs_sample`。
若测试采样器约定每次返回多条固定解，每个条件都收到完整的这些解；不能按条件挑一条并重复。
逐行核对自由变量为 `(spins[:, :-1] * spins[:, -1:] + 1) / 2`，再检查拼回的条件列。

## CIM 与精度

Enterprise 云端采样时，向 KPP 传真实的 `kw.cim.CIMOptimizer` 并用
`PrecisionReducer` 正确包装。注意 `PrecisionReducer` 的导入位置随所依赖的
Enterprise 版本变化：KPP 0.3.0 锁定的 1.3.1 在 `kw.cim`，独立的 SDK 1.4.0 在 `kw.preprocess`。
按实际安装版本导入：

```python
import kaiwu as kw

base = kw.cim.CIMOptimizer(task_name="kpp-skill-user_supplied_task", wait=True)
sampler = PrecisionReducer(   # 从与版本匹配的模块导入
    base, precision=8, truncated_precision=20, target_bits=550,
    only_feasible_solution=False,
)
```

license/checkpoint 设置遵循 Enterprise 精度规则，不要编造云端配置。

## 模型专属约束

`UnsupervisedDBN` 使用前必须先 `create_rbm_layer(input_dim)`，在
`forward`/`transform` 前必须 `mark_as_trained()`。`QVAE` 的 encoder/decoder/bm/sampler
要么构造时注入，要么由子类的四个 `_create_*` 工厂创建；既未注入也无工厂创建的组件
静默为 `None`，直到 forward 才报 `'NoneType' object is not callable`。
QDiffusion 需要提供 proposal 模型、`EnergyModel` 实现、`SequenceTokenSpec`
与能量适配器；使用仓库 `example/qdiffusion` 的工作流，不要发明 backbone
专属接口。

`FeatureSelectionWrapper(model, feature_dim, ...)` 用 `fit_weights(loader, loss_fn, optimizer,
train_epochs)` 训练，optimizer 作用于 `sel.model.parameters()`。`mask_update_epochs=None`
时只在手动 `update_mask(...)` 时刷新掩码。`solver="local_search"` 不需要 license，
`"sa"`/`"kaiwu_cim"` 需要。`update_mask` 的耗时主要在 QUBO 求解，随特征数快速增长，
`hessian_mode`/`max_samples` 只影响导数计算部分。

KPP 的用量统计默认对经其采样触达的 SDK SA/CIM 调用开启。若任务要求关闭，
显式调用 `kaiwu.torch_plugin.disable_usage_stats()` 并说明这一变更。
