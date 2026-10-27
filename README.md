# KPP Skill 安装与使用

此目录是可独立交付的完整 Skill，包含操作说明、参考示例、助手显示信息和环境验证脚本，不需要安装或复制 `benchmark-kpp`。

## 安装

将整个 `kpp-coding-skill` 文件夹导入所用助手的 Skill 目录，保留文件名和目录结构；
安装后可明确要求助手使用 `kpp-skill`，支持自动选择 Skill 的助手也可按任务选择它；
不要只复制 `SKILL.md`，它引用的 `references/` 和 `scripts/` 必须一起保留。

## 使用条件

Skill 文件本身无需安装 Python 依赖，实际运行 KPP 模型时，需要项目的 Python 环境和
`kaiwu-torch-plugin`；已经验证的 KPP 0.3.0 依赖 Kaiwu 1.3.1、PyTorch 2.7.0、
NumPy 2.2.6，Kaiwu 1.3.1 只支持 Python 3.10；已有项目不兼容时使用独立的 3.10 环境，按[安装说明](references/kaiwu_pytorch_plugin.md)处理。

在安装后的 Skill 目录中，用目标项目的 Python 检查：

```sh
python scripts/verify_kpp.py --imports-only
```

设置环境变量 `KAIWU_USER_ID` / `KAIWU_SDK_CODE` 后，可检查小规模本地采样与训练
（脚本会用它们调用 `kaiwu.license.init`，每个环境一次即可）：

```sh
python scripts/verify_kpp.py
```

未找到 license 时脚本输出 `KPP_LICENSE_MISSING` 并以退出码 2 结束，不会等待控制台输入。

只完成导入检查不代表已经验证训练。账号和授权信息由客户在本地配置，Skill 中不包含这些信息。

完整操作说明见 [SKILL.md](SKILL.md)。
