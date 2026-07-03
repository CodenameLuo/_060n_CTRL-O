# CTRL-O：语言可控的对象中心视觉表示学习

*CVPR 2025* | **[项目网站](https://ctrl-o-paper.github.io/)**

> 对象中心表示学习旨在将视觉场景分解为称为“槽位”（slots）或“对象文件”（object files）的固定大小向量，其中每个槽位捕获一个不同的对象。CTRL-O 引入了基于语言的控制，使定向对象提取和多模态应用成为可能，并在文本到图像生成、视觉问答等下游任务上取得了强劲结果。

![CTRL-O 演示](../../images/vg_demo.png)

我们的代码基于 [Object Centric Learning Framework](https://github.com/amazon-science/object-centric-learning-framework)。

## Object Centric Learning Framework（OCLF）

[![Linting and Testing Status](https://github.com/amazon-science/object-centric-learning-framework/actions/workflows/lint_and_test.yaml/badge.svg?branch=main)](https://github.com/amazon-science/object-centric-learning-framework/actions/workflows/lint_and_test.yaml)
[![Docs site](https://img.shields.io/badge/docs-GitHub_Pages-blue)](https://amazon-science.github.io/object-centric-learning-framework/)


## 什么是 OCLF？

OCLF（Object Centric Learning framework，对象中心学习框架）是一个旨在简化对象中心学习研究实验运行的框架，但并不局限于这一用途。其核心思想是：虽然代码通常不具备良好的可组合性，但机器学习中的许多实验非常相似，只存在少量改动，并且这些改动往往只是局部变化。

一个这样的例子是多任务训练：模型可能会被训练为同时解决多个任务。对该模型的不同消融实验会包含不同的模型组件，但整体上仍大体相同。

OCLF 允许在不创建重复代码的情况下完成这类消融实验：它通过配置文件定义模型和实验，并借助 [hydra](https://hydra.cc/) 在配置空间中组合它们。


## 快速开始 - 开发环境设置

安装 OCLF 至少需要 python3.8。安装可以使用 [poetry](https://python-poetry.org/docs/#installation) 完成。安装 `poetry` 后，克隆仓库并设置开发环境：

```bash
git clone git@github.com:dido1998/CTRL-O.git
cd CTRL-O
# 检查 poetry 配置：`poetry config --list`
# 修改 venv 位置（默认是项目根目录 /venv）：`poetry config virtualenvs.path /your/custom/path`
poetry self update
pip install --upgrade pip
poetry install
```

这会在由 poetry 管理的虚拟环境中安装 `ocl` 包，以及用于运行实验的 CLI 脚本。

接下来需要准备数据集。为此，请按照下面的步骤安装数据集转换和创建所需的依赖。


## 数据集

我们提供了用于训练 CTRL-O 的预整理数据集。

1. VG + COCO: https://huggingface.co/adidolkar123/visual_genome_coco
2. VG: https://huggingface.co/adidolkar123/visual_genome/

使用以下命令下载这些数据集：

```
huggingface-cli download <dataset_name> --local-dir scripts/datasets/outputs/ --local-dir-use-symlinks False
```

我们也提供了用于创建你自己的数据集的脚本：

对于 coco 数据集：

```bash
cd scripts/datasets
poetry install
bash download_scripts/download_coco_data.sh
bash download_and_convert.sh COCO
```

这会在路径 `scripts/datasets/outputs/coco` 中创建一个 webdataset。

为了运行实验，需要将数据集暴露给 OCLF：

```bash
cd ../..   # 回到根目录
export DATASET_PREFIX=scripts/datasets/outputs  # 暴露数据集路径
```

## 训练

论文中的主模型在 VG+COCO 数据上训练。要为这次训练运行启动一个实验，可以使用：

```bash
poetry run ocl_train +experiment=projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep
```

这次运行应当达到约 60% 的 binding hits。

训练运行的输出应当存储在 `outputs/projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep/<timestamp>`。

如需更详细的 OCLF 安装、设置和使用指南，请查看文档中的教程。

## 推理和可视化

我们还为 `ocl/cli/inference.py` 中的预训练模型提供了推理和可视化脚本。

运行脚本前，请确保在[此处](../../ocl/cli/inference.py#L32)更新预训练模型 checkpoint 的路径，并在[此处](../../ocl/cli/inference.py#L203)更新你想用于推理的图像路径。

```bash
poetry run python ocl/cli/inference.py
```

## 预训练模型

我们在 Hugging Face 上提供了一个预训练 CTRL-O 模型。你可以使用以下命令下载它：

```bash
huggingface-cli download adidolkar123/pretrained_coco_vgcoco --local-dir pretrained_models/ctrlo --local-dir-use-symlinks False
```

这会将模型 checkpoint 和配置文件下载到 `pretrained_models/ctrlo` 目录。下载后，请更新 `ocl/cli/inference.py` 中的路径，使其指向下载得到的文件。

## 引用

如果你在自己的工作中使用 CTRL-O，请引用下面的 BibTeX 条目：

```bibtex
@inproceedings{didolkar2025ctrlo,
    title={CTRL-O: Language-Controllable Object-Centric Visual Representation Learning},
    author={Didolkar, Aniket Rajiv and Zadaianchuk, Andrii and Awal, Rabiul and Seitzer, Maximilian and Gavves, Efstratios and Agrawal, Aishwarya},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2025}
}
```

## 致谢

本项目是 Max Horn、Maximilian Seitzer、Andrii Zadaianchuk、Zixu Zhao、Dominik Zietlow、Florian Wenzel 和 Tianjun Xiao 开发的 [Object Centric Learning Framework（OCLF）](https://github.com/amazon-science/object-centric-learning-framework) 的一个 fork。

CTRL-O 通过为对象中心表示学习引入基于语言的控制来扩展 OCLF，从而支持特定对象定位和多模态应用。

原项目采用 Apache-2.0 许可证。
