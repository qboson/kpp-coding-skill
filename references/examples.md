# KPP 模型范式

按问题所需能力选择对应范式，不按示例顺序或模型大小选型。
BM/RBM 展示训练函数；QVAE 与 QDiffusion 是集成片段，不能直接作为完整应用交付。
数据、规模与训练配置由任务决定，例子只说明接口关系，不保证业务效果。
复用任务已有环境和兼容采样器；选择 SDK SA/CIM 时才需要对应授权。

## 1. BM 与 RBM：按问题选择

直接刻画二元观测变量的两两相互作用时，可以使用 BM；需要隐藏变量解释观测模式或
要求可见/隐藏二部结构时，可以使用 RBM。是否适合仍由任务效果验证，不要求先训练其中一种。
以下 `records` 是任务提供的二维 0/1 浮点张量；`sampler` 遵守
`solve(ising_matrix) -> [sample_count, num_nodes + 1]` 的 ±1 自旋约定。

### BM：正相直接使用观测变量

```python
from torch.optim import SGD
from kaiwu.torch_plugin import BoltzmannMachine


def train_bm(records, sampler, num_steps, learning_rate):
    """Learn pairwise interactions among the observed binary variables."""
    model = BoltzmannMachine(num_nodes=records.shape[1], device=records.device)
    optimizer = SGD(model.parameters(), lr=learning_rate)
    for _ in range(num_steps):
        optimizer.zero_grad()
        negative = model.sample(sampler).to(records.device)
        loss = model.objective(records, negative)
        loss.backward()
        optimizer.step()
    return model
```

### RBM：正相包含观测变量与隐藏变量

```python
from torch.optim import SGD
from kaiwu.torch_plugin import RestrictedBoltzmannMachine


def train_rbm(records, sampler, num_hidden, num_steps, learning_rate):
    """Learn a visible/hidden joint model when the task needs hidden factors."""
    model = RestrictedBoltzmannMachine(
        num_visible=records.shape[1], num_hidden=num_hidden, device=records.device)
    optimizer = SGD(model.parameters(), lr=learning_rate)
    for _ in range(num_steps):
        optimizer.zero_grad()
        positive = model.get_hidden(records, bernoulli=True)
        negative = model.sample(sampler).to(records.device)
        loss = model.objective(positive, negative)
        loss.backward()
        optimizer.step()
    return model
```

BM 的样本宽度等于观测变量数；RBM 的正负相及采样结果均为可见+隐藏联合状态，
生成观测数据时取 RBM 样本的可见部分。正负相行数可以不同，宽度必须匹配。
`sample` 已完成参考自旋归一化、去尾列与 0/1 转换，不再次删列或转换。
采样本身不提供梯度，`objective` 和 `backward` 保留梯度计算；
核对负相二值性、损失与梯度有限、参数实际更新，并单独检查业务效果和保存后复用。

## 2. KPP QVAE 最小范式

覆盖：子类化 `QVAE` 并实现 `_create_encoder/_create_decoder/_create_bm/
_create_sampler` 四工厂、config 字段、可选均值/偏置校正、encoder+decoder 与 BM 的双优化器。

```python
from kaiwu.torch_plugin import QVAE, RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer

config = SimpleNamespace(num_latent_units=8, loss_type="bernoulli",
                         dist_beta=10.0, kl_beta=1e-6, weight_decay=0.01,
                         sampler_type="sa")

class TinyQVAE(QVAE):
    def _create_encoder(self):
        return nn.Sequential(nn.Linear(self._input_dimension, 32), nn.Tanh(),
                             nn.Linear(32, self._latent_dimensions))
    def _create_decoder(self):
        return nn.Sequential(nn.Linear(self._latent_dimensions, 32), nn.Tanh(),
                             nn.Linear(32, self._input_dimension))
    def _create_bm(self):
        n_vis = self._latent_dimensions // 2
        return RestrictedBoltzmannMachine(num_visible=n_vis,
                                          num_hidden=self._latent_dimensions - n_vis)
    def _create_sampler(self, sampler_type=None):
        return SimulatedAnnealingOptimizer(alpha=0.99)

model = TinyQVAE(INPUT_DIM, nn.Tanh(), config)
mean = training_data.mean(dim=0)  # 可选：逐维校正；不使用时省略这三行
model.set_dataset_mean(mean)
model.set_train_bias(mean)  # 后设置均值不会自动同步偏置
vae_opt = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()))
bm_opt = torch.optim.Adam(model.bm.parameters())
# 每步：recon_x, posterior, q, _ = model(xb)
#   vae: loss = model.loss(xb, recon_x, posterior)
#   bm:  bm_loss = model.bm_loss(q.detach(), config.weight_decay)
# 原生随机重建评估：model.eval() 后，在 torch.no_grad() 内
#   logits, _, _, _ = model(evaluation_data)
#   accuracy = ((torch.sigmoid(logits) > 0.5) == evaluation_data).float().mean()
# forward 会保留输入均值校正与输出偏置；保存的 logits 与指标来自同一次最终评估。
```

`q` 是编码器 logits。`torch.sigmoid(q)` 是底层二值变量的伯努利均值；
`(q > 0).float()` 是硬判定，不能称为均值，也不能替代要求的随机重建。
KPP 0.3.0 的 `QVAE.posterior` 使用 `MixtureGeneric`，解码的是平滑连续变量。
若明确做该后验的确定性均值评估，令 `beta = config.dist_beta`，其均值为
`m0 + (1 - 2*m0) * torch.sigmoid(q)`，其中 `m0 = 1/beta - 1/expm1(beta)`（正 beta）。
输入编码仍须应用训练时的均值校正，解码后仍须加 `_train_bias`。
核实当前后验类型后再选择均值；直接用原生 `model(evaluation_data)` 可避免自行推导。

## 3. KPP QDiffusion 免 checkpoint 范式

覆盖：proposal（含位置嵌入与上下文交互）+ `EnergyModel.score_conditioned` +
`SequenceTokenSpec`（从 `.qdiffusion` 子包导入）+ `objective({"targets":...})`
只训 energy 侧 + `generate`。此片段假设 proposal 已完成独立补全训练；
从零初始化时，先用掩码位置交叉熵训练 proposal，再执行下面的冻结构造。
`V` 是包含特殊 token 的总词表大小，`MASK/PAD/BOS/EOS` 都应小于 `V`。

```python
from kaiwu.torch_plugin import EnergyModel, QDiffusion, QDiffusionConfig
from kaiwu.torch_plugin.qdiffusion import SequenceTokenSpec

class TinyProposal(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 32); self.pos = nn.Embedding(L, 32)
        self.context = nn.TransformerEncoderLayer(32, nhead=4, batch_first=True, dropout=0.0)
        self.out = nn.Linear(32, V)
    def forward(self, tokens):
        h = self.emb(tokens) + self.pos(torch.arange(tokens.size(1), device=tokens.device))
        return self.out(self.context(h))

class PrefEnergy(EnergyModel):
    def __init__(self):
        super().__init__(); self.W = nn.Parameter(torch.zeros(L, V))
    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        onehot = F.one_hot(candidate_tokens, V).float()
        energy = -(onehot * self.W).sum(dim=-1)
        return (energy * attention_mask).sum(dim=-1, keepdim=True)  # [B, 1]

generator = QDiffusion(proposal_model=proposal, energy_model=PrefEnergy(),
    token_spec=SequenceTokenSpec(mask_id=MASK, pad_id=PAD, bos_id=BOS, eos_id=EOS),
    config=QDiffusionConfig(num_diffusion_timesteps=50, num_candidates=4,
        proposal_temperature=1.0, energy_temperature=1.0),
    freeze_proposal=True)
# 训练：outputs = generator.objective({"targets": batch}); loss = outputs["energy_objective"].mean()
# 生成：samples = generator.generate(torch.full((B, L), MASK, dtype=torch.long), max_steps=20)
```
