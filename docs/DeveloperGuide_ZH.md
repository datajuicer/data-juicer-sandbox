# 开发者指南

正如上一章节所说，开发者可开发更多的可配置组件并将它们添加到对应的工厂类中，并用参数`type`进行实例化方法分配。实现了组件后，开发者可以将它们封装为钩子，并将钩子注册到工作列表中，工作列表在流水线中进行编排后，沙盒流水线执行时，会依次在每个步骤执行每个工作列表中的工作。这其中的每一个部分：组件、组件工厂、钩子、工作列表、流水线注册与执行流程编排，都可以由开发者自定义。各个部分的关系由下图示意。
![sandbox-pipeline](https://img.alicdn.com/imgextra/i3/O1CN01ERmGre1uz3luKOn4n_!!6000000006107-2-tps-4655-1918.png)

## 组件内部实现

目前组件主要分为三个大类：

- **执行器（Executor）**：由于数据执行器已经由Data-Juicer的Executor承担，因此此处的执行器特指模型的执行器，包括模型训练、推理、评估等执行器。代码位于 [这里](https://github.com/datajuicer/data-juicer-sandbox/blob/main/data_juicer_sandbox/model_executors.py)
- **评估器（Evaluator）**：用于对数据集或者模型进行质量以及性能的评估。代码位于 [这里](https://github.com/datajuicer/data-juicer-sandbox/blob/main/data_juicer_sandbox/evaluators.py)
- **数据池操作器（DataPoolManipulator）**：用于操作数据池，例如构建、组合、采样等。代码位于 [这里](https://github.com/datajuicer/data-juicer-sandbox/blob/main/data_juicer_sandbox/data_pool_manipulators.py)

### 执行器

模型执行器核心功能为对配置文件中指定的模型用指定的数据集进行训练、推理或评测。模型执行器需继承`BaseModelExecutor`并实现若干核心方法：

- 模型执行器的具体行为（训练、推理、评测等）需要在`_run`方法中进行定义
- 模型执行器无返回值，执行时需要进行监测的关键指标通常从模型执行产出的日志中进行解析（如loss、评测结果等），解析与监测过程需要由`_watch_run`方法定义
- 模型执行器在执行时需要数据集输入，因此需要实现`data_connector`方法将数据集由沙盒中的格式转为该模型依赖的框架或者模型库所需要的格式

需要注意的是，为了在模型训练过程中及时监控训练指标（如loss）的变化，需要同时对训练时产生的日志进行监控。因此，执行模型训练的`_run`方法以及监控日志的`watch_run`方法都需要为异步方法，即被关键字`async`修饰。在`run`方法中，我们在训练开始前将标准输出流（stdout）和标准错误流（stderr）都重定向到指定的日志文件，并创建两个异步任务分别执行上述两个方法，它们分别进行以下任务：

- `_run`方法：读入数据集后，根据模型训练配置开始进行模型训练，训练结束后向标准输出流（已重定向到指定的日志文件）输出一个预定义的任务执行结束标识符
- `watch_run`方法：监控指定的日志文件，逐行读取，并调用根据模型训练框架自定义的`_watch_run`方法解析最新的日志内容行，提取关键指标并进行监测，直到读取到预定义的任务结束标识符

### 评估器

评估器核心功能为对待评估对象使用某种方法进行质量、性能等维度的评估，并最终返回一个评估结果，通常为数值型结果。评估器需继承基类`BaseEvaluator`并实现`run`方法。`run`方法默认接受两个必要参数：

- `eval_type`：评估类型，用于在某种评估器内部进行评估类型选择
- `eval_obj`：待评估的对象

用户也可根据自己的实现方式对这两个参数进行扩展使用。

### 数据池操作器

数据池操作器的核心功能是操作数据池，例如构造、组合、采样等。数据池操作器需要继承自基类 `BaseDataPoolManipulator`，并实现 `run` 方法。`run` 方法所需的参数通常来自 `__init__` 方法中的输入数据池配置，涵盖输入数据池、导出路径以及每种操作器的具体参数。

用户可以参考 [这里](https://github.com/datajuicer/data-juicer-sandbox/blob/main/data_juicer_sandbox/data_pool_manipulators.py) 每种操作器的 `run` 方法的 doc string 了解更多详细信息。

## 流水线钩子

正如章节开始部分所说，在流水线中，我们需要实现若干钩子将组件与流水线执行步骤通过工作列表连接起来。被激活的钩子会在流水线的工作列表中进行注册，然后在流水线执行时依次对各个步骤工作列表中的钩子执行。四个步骤对应的工作列表分别如下：

1. **数据/模型洞察**：洞察工作列表 -- probe_jobs
2. **基于洞察结果的数据菜谱微调迭代**：菜谱微调工作列表 -- refine_recipe_jobs
3. **数据处理与模型训练**：执行工作列表 -- execution_jobs
4. **数据/模型评估**：评估工作列表 -- evaluation_jobs

通常情况下，我们只需要为一类组件工厂实现一种钩子函数即可。而除了依赖于组件的钩子外，还有一些依赖于Data-Juicer已有功能或工具以及其他第三方库的钩子。这些钩子与依赖的组件、工具以及工作列表的对应关系如下：

| 钩子                                 | 功能                                | 依赖的组件工厂                                                               | 依赖的工具或库                                                                                                                                           | 注册工作列表                                          |
|------------------------------------|-----------------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| `ProbeViaAnalyzerHook`             | 分析与洞察数据集质量、多样性等维度分布               | 数据分析工厂（DataAnalyzerFactory）                                           | Data-Juicer分析器Analyzer                                                                                                                            | 洞察工作列表（probe_jobs）<br />评估工作列表（evaluation_jobs） |
| `ProbeViaModelInferHook`           | 分析与洞察数据集对于模型的影响，挖掘与洞察“难”数据与“脏”数据  | 数据处理工厂（DataExecutorFactory）<br />模型数据评估工厂（ModelInferEvaluatorFactory） | Data-Juicer数据处理器Executor                                                                                                                          | 洞察工作列表（probe_jobs）<br />评估工作列表（evaluation_jobs） |
| `GeneralProbeHook`                 | 提供通用的数据集探测能力，包括数据集排序等             | 通用数据探测工厂（GeneralProbeFactory）                                         | -                                                                                                                                                 | 洞察工作列表（probe_jobs）                              |
| `RefineRecipeViaKSigmaHook`        | 根据数据集洞察结果，利用k-sigma方法对数据菜谱超参进行微调  | -                                                                     | Data-Juicer超参优化工具HPO中的k-sigma菜谱微调工具                                                                                                               | 菜谱微调工作列表（refine_recipe_jobs）                    |
| `RefineRecipeViaModelFeedbackHook` | 利用模型洞察与反馈结果对数据菜谱超参进行微调            | TODO                                                                  | -                                                                                                                                                 | 菜谱微调工作列表（refine_recipe_jobs）                    |
| `ProcessDataHook`                  | 基于当前数据菜谱对数据集进行处理与清洗               | 数据处理工厂（DataExecutorFactory）                                           | Data-Juicer数据处理器Executor                                                                                                                          | 执行工作列表（execution_jobs）                          |
| `DataPoolManipulationHook`         | 操作数据池，包括构造，组合，采样等                 | 数据池操作工厂（DataPoolManipulatorFactory）                                   | -                                                                                                                                                 | 执行工作列表（execution_jobs）                          |
| `GeneralDataExecutorHook`          | 通用数据集处理能力，包括格式转换等                 | 通用数据处理工厂（GeneralDataExecutorFactory）                                  | -                                                                                                                                                 | 执行工作列表（execution_jobs）                          |
| `TrainModelHook`                   | 基于当前数据集训练一个模型                     | 模型训练工厂（ModelTrainExecutorFactory）                                     | [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) <br/> [InternVL](https://internvl.readthedocs.io/en/latest/index.html)                       | 执行工作列表（execution_jobs）                          |
| `InferModelHook`                   | 模型基于给定输入让模型产生输出                   | 模型推理工厂（ModelInferExecutorFactory）                                     | [EasyAnimate](https://github.com/aigc-apps/EasyAnimate)                                                                                              | 执行工作列表（execution_jobs）                          |
| `EvaluateDataHook`                 | 对当前数据集进行数据质量等维度的评估                | 数据评估工厂（DataEvaluatorFactory）                                          | 图像或视频的[inception metrics](../tools/mm_eval/inception_metrics/README_ZH.md)，如FID、FVD <br /> [VBench](../tools/mm_eval/vbench_metrics/README_ZH.md) | 评估工作列表（evaluation_jobs）                         |
| `EvaluateModelHook`                | 对当前训练后的模型进行评估                     | 模型评估工厂（ModelEvaluatorFactory）                                         | [InternVL COCO Caption](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html#evaluating-the-fine-tuned-model)           | 评估工作列表（evaluation_jobs）                         |

值得注意的是，一个钩子可以在多个工作列表进行注册，因为这个钩子在不同的流水线阶段可以扮演不同的角色，比如我们可以对处理前后的数据集都进行分析，以比较数据集处理前后的质量、多样性等维度的变化情况。

## 自定义沙盒流水线

用户直接在参数配置文件中修改任务配置列表即可实现任务修改和编排。

## 监测器

在上述章节中，反复提到“监测”这个概念。流水线会对各个步骤中产生的若干指标都进行监测，这些监测过程都依靠沙盒监测器`SandboxWatcher`实现的。

`SandboxWatcher`基于wandb实现，主要包括4个方法：

- `setup_sweep`：在多轮HPO模式下会调用，多轮HPO由wandb中的sweep支持，因此需要额外传入`hpo_config`配置文件对其进行初始化
- `watch_cfgs`：对sandbox实验以及各个组件的配置文件进行监测与更新
- `watch`：对某个具体指标或实验结果进行监测，并记录到wandb日志
- `query`：对某个具体指标或实验结果从wandb日志中进行查询

## 上下文信息实现细节

`context_infos` 包含两个级别：

- pipeline 级别：它是 `context_infos` 的第一级，它是一个字典，键是 pipeline 名称，值是该 pipeline 中每个 job 的上下文信息列表。
- job 级别：它是 `context_infos` 的第二级，它是一个字典列表，每个字典代表特定 job 的上下文信息，其中 `meta_name` 用于标识 job，其他键值对的键是该 job 的输出名称，值是输出值。

以下是 `context_infos` 的一个示例：

```python
{
    'pipeline_0': [
        {
            'meta_name': 'name1',
            'res1_key': 'res1_value',
            'res2_key': <res2_value>,
        },
        {
            'meta_name': 'name2',
            'res3_key': 'res3_value',
        },
    ],
    'pipeline_1': [
        {
            'meta_name': 'name3',
            'res4_key': <res4_value>,
        },
    ],
    'pipeline_2': [
        {
            'meta_name': 'name4',
            'res5_key': ['res5_1_value', 'res5_2_value'],
        },
    ],
}
```

## 环境管理器

Sandbox 支持不同类型的第三方库，用于训练、评估等。如果将它们全部放在一个环境中，一些重要且复杂的依赖项可能会发生版本冲突。因此，我们提供了一个易于使用的环境管理器，用于将不同第三方库在不同环境中分别进行管理，允许用户在独立的环境中运行命令。

环境的基本类是 `Env`，位于 [这里](https://github.com/datajuicer/data-juicer-sandbox/blob/main/data_juicer_sandbox/env_manager.py)，其实现如下：

```python
class Env(ABC):
  
    @abstractmethod
    def create(self):
        """
        创建一个环境
        """
        raise NotImplementedError(
            'This method must be implemented in subclass.')

    @abstractmethod
    def check_availability(self):
        """
        检查环境管理器的可用性
        """
        raise NotImplementedError(
            'This method must be implemented in subclass.')

    @abstractmethod
    def exists(self):
        """
        检查环境是否存在
        """
        raise NotImplementedError(
            'This method must be implemented in subclass.')

    @abstractmethod
    def install_py_deps(self):
        """
        安装 Python 依赖项
        """
        raise NotImplementedError(
            'This method must be implemented in subclass.')

    @abstractmethod
    def run_cmd(self):
        """
        在该环境中运行命令
        """
        raise NotImplementedError(
            'This method must be implemented in subclass.')
```

它包含五个主要的抽象方法：
- `create`：如果环境不存在，则创建环境。
- `check_availability`：检查环境管理器（例如 `conda`、`venv`）的可用性。
- `exists`：检查环境是否存在。
- `install_py_deps`：安装 Python 依赖项。通常支持三种方式：通过“requirements.txt”文件路径、依赖项列表、或指向库代码库的目录路径。
- `run_cmd`：在此环境中运行命令。

现在我们提供了两种 `Env` 的具体实现：
- `CondaEnv`：使用 `conda` 或 `mamba` 管理环境。
- `VirtualEnv`：使用 `venv`、`virtualenv` 或 `uv venv` 管理环境。

在初始化环境管理器时，我们可以通过设置配置文件中的 `env_manager` 参数来指定要使用的环境管理器，并通过设置 `env_name` 参数来指定环境的名称。基本用法示例如下：

```python
from data_juicer_sandbox.env_manager import ENV_ROUTER

env_manager = 'conda'
env_name = 'new_conda_env'

# 创建一个环境
env = ENV_ROUTER[env_manager](
    env_name=env_name,
    env_manager=env_manager)
# 检查环境管理器可用性
if not env.check_availability():
    # 该环境管理器不可用
    exit()
# 创建一个新环境。如果环境已存在，则使用已存在的环境
env.create()

# 安装额外的依赖项
# 使用 "requirements.txt" 文件
env.install_py_deps("/path/to/requirements.txt")
# 使用依赖项列表
env.install_py_deps(["torch", "torchvision"])
# 使用指向库代码库的目录路径，如 InternVL
env.install_py_deps("/path/to/a/third-party/library")

# 在该环境中运行一条命令
cmd = "python train.py"
env.run_cmd(cmd)
```

[这里](https://github.com/datajuicer/data-juicer-sandbox/blob/main/data_juicer_sandbox/specific_hooks/intervl_coco_captioning/model_hooks.py) 的 `InternVLCOCOCaptionEvaluator` 类提供了在钩子中使用环境管理器的完整示例。
