---
name: kpp-skill
description: "使用 Kaiwu PyTorch Plugin（KPP）安装、构建与训练 PyTorch 能量模型，并接入 Kaiwu 采样器。适用于 RBM/BM、DBN、使用能量模型先验的 QVAE、能量引导离散序列生成 QDiffusion、为 PyTorch 模型做 MAIFS 特征选择，或 kaiwu.torch_plugin 的调试。明确需要 KPP 或场景适配时主动检查并安装 kaiwu-torch-plugin、处理依赖冲突、验证导入和训练。普通回归、一般神经网络、普通 VAE/扩散模型不触发；不依附 PyTorch 模型的纯 QUBO/Ising 组合优化不属于本 Skill 的范围。"
metadata:
  version: "0.2.6"
---

# KPP 能量模型与采样

负责 PyTorch 模型与 Kaiwu 采样器的集成。模型参数、目标函数和优化器留在 PyTorch；
通过兼容采样器接入 Kaiwu 求解器或任务已有的采样流程。不要把普通机器学习任务自动改写成能量模型任务。

## 选择组件

先根据输入、输出和需要刻画的关系选择结构，说明所选组件如何满足任务要求。
用户指定结构时按指定结构完成；自行选型时，判断任务需要观测变量间的直接关系、
隐藏因素、分层表示、潜变量重建、上下文生成还是特征筛选，再选择对应组件。
模型大小不是默认优先级；先验证能力与效果是否匹配，再在满足要求的方案间比较训练和采样成本。

| 问题需要的能力 | 组件 | 匹配依据 |
|---|---|---|
| 用隐藏因素解释观测模式，或学习可见/隐藏二部结构 | `RestrictedBoltzmannMachine` | 隐藏表示参与任务所需的分布学习；仅有二元输入不足以决定采用 RBM |
| 直接刻画二元节点的两两相互作用，或固定部分节点后采样 | `BoltzmannMachine` | 节点及其相互作用对应问题变量；用任务指标验证是否足以表达所需关系 |
| 通过多层 RBM 学习分层特征 | `UnsupervisedDBN` | 任务需要逐层学习的表示，或用户明确要求 RBM 堆栈 |
| 编码、重建及可采样的能量模型潜变量先验 | `QVAE` | 重建与潜变量先验都服务于任务；普通自编码器不因此改成 QVAE |
| 上下文相关的离散候选生成与能量引导 | `QDiffusion` | proposal 提供所需上下文，energy 对候选进行任务相关的引导 |
| 从已有 PyTorch 模型的输入里选择特征子集 | `FeatureSelectionWrapper` | 任务需要输入特征筛选，使用周期更新的 0/1 掩码 |

二元数据、标签数量或示例出现的先后顺序不能单独决定选型。
用任务要求的关联、重建、补全或预测等结果验证选择；仅导入成功或输出形状正确不代表匹配。
任务需要高阶关系或上下文时，不能仅凭两两统计达标就认定模型足够。
BM 与 RBM 的不同训练结构见 [模型范式](references/examples.md)，只读当前任务相关的部分。

这是同一 KPP 包的不同组件，无需分别安装六个插件。
特征选择虽然内部求解 QUBO，但服务于 PyTorch 模型训练，属于本 Skill；
纯选址、调度、Max-Cut 等不依附模型、直接求最优决策的任务使用 Kaiwu SDK 技能。
若同一任务同时包含组合优化与能量训练，可分别加载两个技能并使用各自解释器。

## 应用任务的最小闭环

1. 用代表性数据检查所选结构的任务效果，并记录真实训练与生成的规模、耗时，据此安排正式运行，
   在总预算中留出模型保存、新进程加载、生成与报告核对的时间。
2. 尽早打通题目要求的完整入口。每次验证后保存当前效果最好且可复用的模型、配置和报告；
   后续试验使用新的输出目录，交付目录始终保留可运行版本。
3. 批量生成按实际行数累计：追加一批时累加该批的行数，达到请求数量后再截取。
   先用小请求核对数量、宽度和取值，批数不等于样本数。
4. 分开检查直接复制原记录、模型采样自然重合和人工筛选。常见组合可能在独立生成中重合；
   过滤重合记录或强制唯一会改变分布，使用后重新核对比例与关联等业务指标。
5. 按任务公开的业务要求验证，保存模型后在新进程中仅用交付模型和请求输入重新生成，
   报告同一次运行的质量与耗时。达到本次目标后完成交付；未达到时保留当前成果并如实说明，
   继续尝试以剩余预算允许完成上述核对为前提。

## 检查、自动安装、验证

1. 复用用户指定/项目已有解释器，核对 Python、`kaiwu-torch-plugin`、`kaiwu`、`torch` 的版本。
   同一环境运行一次 `python -m pip check`，再运行技能目录内
   [scripts/verify_kpp.py](scripts/verify_kpp.py) 的 `--imports-only` 模式。
   已有本次检查结果时直接复用，随后进入任务，不额外扫描环境或试跑通用训练。
2. 缺包时才检查安装计划并安装；依赖冲突、离线包、Python 版本或 wheel 支持有问题时，
   按 [安装说明](references/kaiwu_pytorch_plugin.md) 处理，依据实际元数据选择兼容环境。
3. 复用任务提供的兼容采样器。选择 SDK SA/CIM 时才检查对应授权，
   不把 SDK 授权缺失当成所有本地采样器都不能训练。授权初始化和非交互限制见安装说明。
4. 用实际任务程序验证所选模型；没有任务程序且需要检查 SDK 采样训练时，才运行验证脚本默认模式。
   `KPP_IMPORT_OK` 只证明导入；`KPP_TRAIN_OK` 只证明脚本覆盖的小规模训练。

检查日志、临时文件和验证副本留在任务目录，云端 CIM 仅在任务明确需要且已有配额授权时使用。

## 训练与生成契约

BM 与 RBM 的公开导入路径：

```python
from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine
```

选择 SDK SA 时，`SimulatedAnnealingOptimizer` 从 `kaiwu.classical` 导入。
BM 构造使用 `BoltzmannMachine(num_nodes, quadratic_coef=None, linear_bias=None, device=None)`；
RBM 使用 `RestrictedBoltzmannMachine(num_visible, num_hidden, quadratic_coef=None,
linear_bias=None, device=None)`。学习率、权重衰减等设置交给 PyTorch 优化器，
不向模型构造器传入 `learning_rate`、`k`、`batch_size` 或其他库的训练参数。

- **RBM**：`get_hidden(batch, bernoulli=True)` 和 `sample(sampler)` 返回可见+隐藏联合状态，
  宽度是 `num_visible + num_hidden`。正相批次与负相解池行数可不同，分别求均值；
  不硬凑相同样本数。`objective(positive, negative)` 返回标量。
  直接写 `positive = model.get_hidden(batch, bernoulli=True)`，不解包返回值，也不再拼接可见层。
  `get_hidden` 默认 `bernoulli=False`，隐藏部分为概率；用于 RBM 正相期望合法，
  交付二值联合样本时才需 `bernoulli=True`。不要把概率状态当成已经采出的二值样本。
  解池容量按实际采样器配置，SDK SA 使用 `SimulatedAnnealingOptimizer(size_limit=...)`；
  不编造 `set_config` 或给 `sample` 增加未定义的 `num_solutions` 参数。
- **SDK 采样**：`sampler.solve` 返回含尾部参考自旋的 ±1 二维数组。KPP 的 `sample`
  内部完成参考自旋归一化、删除尾列、转成 0/1 浮点张量。对 `sample` 返回值不要再删列或转换。
  用户提供固定测试解时，每次 `solve` 原样返回整组解；不要把各解分配给不同条件，
  或重复首行凑样本数。条件采样须同时核对固定列和每条解转换后的自由列，不能只检查形状。
- **训练顺序**：每步 `optimizer.zero_grad()` → 正负相 → `objective` → `backward()` →
  `step()`；`objective` 和 `backward()` 不能放在 `torch.no_grad()` 中。`sample()` 返回的是
  无梯度的采样张量，梯度来自 `objective` 重新计算能量，`sample()` 是否在 `no_grad` 内不影响训练。模型、数据、手动张量保持同一 device。
  检查正负相宽度、负相二值性、损失/梯度有限、参数实际更新。
- **BM / DBN**：条件采样用 `condition_sample(sampler, s_visible)`，它固定的总是前 `s_visible.shape[1]` 个节点，
  每个条件行调用一次求解器。各行已知位置不同时逐行重排：按“已知在前”置换 `linear_bias` 和对称化后的
  `quadratic_coef`，用置换后的参数建临时 `BoltzmannMachine` 调用 `condition_sample`，再把列逆置换回原顺序。
  DBN 使用前调用
  `create_rbm_layer(input_dim)`，训练完成后 `mark_as_trained()` 再做 `forward/transform`。
- **QVAE**：可构造注入 encoder/decoder/bm/sampler，也可子类实现四个 `_create_*` 工厂。
  config 含 `num_latent_units / loss_type / dist_beta / kl_beta / weight_decay`。
  伯努利均值校正是可选设置；使用时在构造后分别调用 `set_dataset_mean`、`set_train_bias`，
  不认为后设置均值会自动更新偏置。均值可为标量或逐维张量，以当前接口为准。
  encoder+decoder 与 BM 用两个优化器，
  BM 更新使用 `bm_loss(q.detach(), weight_decay)`；参考 [模型范式](references/examples.md)。
  最终评估保留训练时的输入均值校正与输出偏置。`q` 是编码器 logits，
  `(q > 0).float()` 是硬判定，不是后验均值。原生重建与均值的区别见参考文档的 QVAE 段落。
  随机重建、解码真实后验均值均可；
  从实际最终 logits 计算并保存指标，说明所用方式，不把少量训练的质量当作接口通过条件。
- **QDiffusion**：proposal 输入 [B,L]、输出 [B,L,V] logits。需要上下文补全时，
  用自注意力等结构让缺失位置读取其他位置；逐位置 embedding + MLP 只能学习位置偏好，
  即使符合率很高，也不满足上下文补全。`EnergyModel`
  子类实现 `score_conditioned(noisy, candidate, attention_mask)` 输出 [B,1]。
  `SequenceTokenSpec` 从 `kaiwu.torch_plugin.qdiffusion` 导入，顶层未导出。
  仅给出“词表大小 V”时，保持输出宽度 V；特殊 id 必须落在该词表内并与内容 token 分开。
  不自行把 V 解释为内容词表并扩成 V+4。内容 id 占满 V 或特殊 id 映射不明时，
  在方案中列出待确认的映射；确认前不采用扩大的输出宽度，也不缩减用户已有的内容类别。
  proposal 从零训练时先完成独立补全训练，
  再冻结并训练 energy；原生 `objective` 内部截断 proposal 梯度，不能用它预训练 proposal。
  训练用 `generator.objective({"targets": batch})`，生成用 `generator.generate(init, max_steps=...)`。
  `freeze_proposal=True` 时只优化 energy 侧。完整集成条件见 [模型范式](references/examples.md)。
- **特征选择（MAIFS）**：`from kaiwu.torch_plugin import FeatureSelectionWrapper, QuadraticLinearSolver`。
  构造签名 `FeatureSelectionWrapper(model, feature_dim, lambda_reg=0.0, cardinality_k=None,
  min_selected_features=None, max_selected_features=None, solver="local_search",
  mask_update_epochs=None, input_feature_axis=-1, solver_kwargs=None)`。训练用
  `sel.fit_weights(loader, loss_fn, optimizer, train_epochs)`：loader 产出 `(x, y)`，
  `loss_fn` 返回标量，optimizer 作用于 `sel.model.parameters()`。掩码每 `mask_update_epochs`
  个 epoch 刷新一次；为 `None` 时只在手动调用
  `update_mask(loader, loss_fn, hessian_mode="full"|"diagonal", max_samples=1000)` 时刷新。
  结果用 `selected_indices()`、`num_selected()`、`get_support()` 读取。
  `solver="local_search"` 不需要 license；`"sa"`、`"kaiwu_cim"` 走 Kaiwu SDK，需要 license。
  `update_mask` 的耗时主要在 QUBO 求解与数量投影，随特征数快速增长（本机 400 个特征约 13–15 秒，
  导数计算不到 1 秒）；`hessian_mode` 和 `max_samples` 只影响导数部分，按实际特征数预留时间。
  `QuadraticLinearSolver().solve(Q, b)` 只把 QUBO 项转成带一个辅助自旋的 Ising 矩阵，不负责求解。

## 交付与验收

新建任务交付完整脚本，补齐数据、import、配置、训练循环与结果输出；改已有代码时只改任务所需部分。
只要求方案或环境记录时，实际保存指定文档；多轮需求补齐后更新交付，并核对文件已写入。
参考文档中的 QVAE/QDiffusion 片段是结构示意，含上下文变量，不能冒充可直接运行的脚本。
API 不确定时定位当前 `kaiwu.torch_plugin.__file__`，查实际签名和源码。
报告真实解释器、依赖版本和执行结果；RBM 通过不能代表其他模型、GPU 或云端 CIM 也通过。
