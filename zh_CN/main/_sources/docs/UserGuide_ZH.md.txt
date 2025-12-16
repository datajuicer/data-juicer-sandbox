# 用户指南

## 什么是沙盒实验室（DJ-Sandbox）？

Data-Juicer 中的 DJ-Sandbox 是一个连接数据和模型反馈的中间件，能够在各种任务中实现高性能和低成本的验证。它旨在为用户提供持续增强数据模型方案的最佳实践，具有低开销、可移植性和指导性等特点。在沙盒中，用户可以基于小规模数据集和模型快速实验、迭代和优化方案，然后迁移到更大尺度上，以生成高质量数据，服务于大规模模型。

除了 Data-Juicer 提供的基本数据优化和方案优化功能外，用户还可以无缝使用可配置组件，例如数据探测和分析、模型训练和评估以及基于数据和模型反馈的方案优化，从而形成数据模型研发的最佳流水线。

更多详细信息，请参阅我们的[论文](http://arxiv.org/abs/2407.11784)（ICML'25 Spotlight）。


## 应用
我们将沙盒应用于到了众多前沿模型，例如 Mini-Gemini 和 InternVL-2.0（两个受 LLaVA 启发的图像转文本生成模型），EasyAnimate 和 T2V-Turbo（两个基于 Diffusion Transformer 的文本转视频生成模型），以及一个用于图文预训练的 CLIP 模型。在此之上，我们曾在[VBench](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)文生视频排行榜取得了新的榜一。
![top-1_in_vbench](https://img.alicdn.com/imgextra/i1/O1CN01I9wHW91UNnX9wtCWu_!!6000000002506-2-tps-1275-668.png)

相关模型已在ModelScope和HuggingFace平台发布，训练模型的数据集也已开源。

| 开源模型或数据集 | 链接 | 说明 |
| ------------ | --- | --- |
| Data-Juicer (T2V, 147k) |  [ModelScope](https://modelscope.cn/models/Data-Juicer/Data-Juicer-T2V) <br> [HuggingFace](https://huggingface.co/datajuicer/Data-Juicer-T2V) | 对应榜单中 Data-Juicer (T2V-Turbo) 模型 |
| Data-Juicer (DJ, 228k) | [ModelScope](https://modelscope.cn/models/Data-Juicer/Data-Juicer-T2V-v2) <br> [HuggingFace](https://huggingface.co/datajuicer/Data-Juicer-T2V-v2) | 对应榜单中 Data-Juicer (2024-09-23, T2V-Turbo) 模型 |
| data_juicer_t2v_optimal_data_pool | [Aliyun](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/data_juicer_t2v_optimal_data_pool.zip) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/data-juicer-t2v-optimal-data-pool)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/data-juicer-t2v-optimal-data-pool) | Data-Juicer (T2V, 147k) 的训练集 |
| data_juicer_t2v_evolution_data_pool | [Aliyun](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/data_juicer_t2v_optimal_data_pool_s2.zip) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/data-juicer-t2v-evolution-data-pool) | Data-Juicer (2024-09-23, T2V-Turbo) 的训练集 |

Data-Juicer (DJ, 228k)模型输出样例如下表所示。
  | 文本提示 | 生成视频 |
  | --- | --- |
  | A beautiful coastal beach in spring, waves lapping on sand, zoom out | [![Case 0](https://img.alicdn.com/imgextra/i1/O1CN01KuJeOE1Ylqnk9zYkc_!!6000000003100-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case0.mp4) |
  | a boat accelerating to gain speed | [![Case 1](https://img.alicdn.com/imgextra/i2/O1CN01i1iMFE1TKlIUlqE8d_!!6000000002364-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case1.mp4) |
  | A boat sailing leisurely along the Seine River with the Eiffel Tower in background by Hokusai, in the style of Ukiyo | [![Case 2](https://img.alicdn.com/imgextra/i2/O1CN01u2cjJE1RBwRFeCFuo_!!6000000002074-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case2.mp4) |
  | a bottle on the left of a wine glass, front view | [![Case 3](https://img.alicdn.com/imgextra/i4/O1CN01vdMm6Q1xWc1CoJZW6_!!6000000006451-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case3.mp4) |
  | A corgi's head depicted as an explosion of a nebula | [![Case 4](https://img.alicdn.com/imgextra/i2/O1CN014oPB8Q1IrJg0AbUUg_!!6000000000946-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case4.mp4) |
  | A graceful ballerina doing a pirouette on a dimly lit stage, with soft spotlight highlighting her movements. | [![Case 5](https://img.alicdn.com/imgextra/i4/O1CN01yNlsVu1ymvkJgkvY8_!!6000000006622-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case5.mp4) |

复现论文实验请参考下面的sandbox使用指南，下图的实验流程，[初始数据集](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/data_juicer_t2v_init_data_pool.zip)，以及该流程的工作流的配置文件demo： [1_single_op_pipeline.yaml](https://github.com/datajuicer/data-juicer-sandbox/tree/main/configs/easyanimate_text_to_video/1_single_op_pipeline.yaml) 、[2_multi_op_pipeline.yaml](https://github.com/datajuicer/data-juicer-sandbox/tree/main/configs/easyanimate_text_to_video/2_multi_op_pipeline.yaml)、[3_duplicate_pipeline.yaml](https://github.com/datajuicer/data-juicer-sandbox/tree/main/configs/easyanimate_text_to_video/3_duplicate_pipeline.yaml)。
![bench_bottom_up](https://img.alicdn.com/imgextra/i2/O1CN01xvu2fo1HU80biR6Q5_!!6000000000760-2-tps-7756-3693.png)

## 快速上手

### 依赖准备

在使用沙盒实验室前，你可能需要使用如下命令安装沙盒：

```shell
git clone https://github.com/datajuicer/data-juicer-sandbox.git
cd data-juicer-sandbox/
uv pip install -e ".[all]"
```

并根据官方说明准备好沙盒中使用的第三方库（例如 EasyAnimate 、 VBench 、 InternVL 等），或者您也可以简单地从 GitHub 克隆第三方存储库，并在沙盒运行期间将安装过程留给我们的 `EnvManager` 完成。

**注意**：一些沙盒的依赖还需要额外的领域依赖。

1. 要使用[ModelScope](https://github.com/modelscope/modelscope)时需从ModelScope的独立host安装其相关依赖：
```shell
uv pip install "modelscope[framework,nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
可参考[ModelScope文档](https://www.modelscope.cn/docs/intro/environment-setup)获取更多信息。

2. 要使用[EasyAnimate](https://github.com/aigc-apps/EasyAnimate)时需要执行如下安装脚本：
```shell
cd thirdparty/models/
bash setup_easyanimate.sh
cd ../../
```

如果使用沙盒过程中，这些第三方依赖抛出了一些"未找到模块（Module-Not-Found）"的报错时，用户需要先检查这些库的文档以寻求帮助。

### 准备沙盒配置文件

沙盒实验总共会依次执行四类任务：数据/模型洞察（`probe_job_configs`）、基于洞察结果的数据菜谱微调迭代（`refine_recipe_job_configs`）、数据处理与模型训练（`execution_job_configs`）和数据/模型评估（`evaluation_job_configs`）。每类任务中，任务按照配置的任务列表依次执行。每个任务需要指定：挂载这个任务的钩子（`hook`），用于识别钩子的标记名(`meta_name`)，Data-Juicer数据处理参数（`dj_configs`），以及该任务其他的特定参数（`extra_configs`）。这些参数中`hook`是必须指定的，其他允许置空。`dj_configs`可以参考完整的Data-Juicer数据处理参数 [config_all.yaml](https://github.com/datajuicer/data-juicer/blob/main/data_juicer/config/config_all.yaml)。`extra_configs`为任务特定的参数，没有限定，可以是模型训练、推理、评测等参数，比如用`path_k_sigma_recipe`指定利用k-sigma方法微调后的数据菜谱保存路径。一个sandbox的配置文件示例可参考`configs/demo/sandbox/sandbox.yaml`：

```yaml
# Sandbox config example

# global parameters
project_name: 'demo-sandbox'
experiment_name: 'demo-sandbox-run0'              # for wandb tracer name
hpo_config: null                                  # path to a configuration file when using auto-HPO tool.

# configs for each job, the jobs will be executed according to the order in the list
probe_job_configs:
  - hook: 'ProbeViaAnalyzerHook'
    meta_name: 'analysis_ori_data'
    dj_configs: 'configs/demo/process.yaml'
    extra_configs:

refine_recipe_job_configs:
  - hook: 'RefineRecipeViaKSigmaHook'
    meta_name: 'analysis_ori_data'
    dj_configs: 'configs/demo/process.yaml'
    extra_configs:
      path_k_sigma_recipe: './outputs/demo-process/k_sigma_new_recipe.yaml'

execution_job_configs:
  - hook: 'ProcessDataHook'
    meta_name:
    dj_configs: './outputs/demo-process/k_sigma_new_recipe.yaml'
    extra_configs:
  - hook: 'TrainModelHook'
    meta_name:
    dj_configs:
    extra_configs: 'configs/demo/gpt3_extra_train_config.json'

evaluation_job_configs:
  - hook: 'ProbeViaAnalyzerHook'
    meta_name: 'analysis_processed_data'
    dj_configs: 'configs/demo/process.yaml'
    extra_configs:
  - hook: 'EvaluateDataHook'
    meta_name: 'eval_data'
    dj_configs:
    extra_configs: 'configs/demo/gpt3_data_quality_eval_config.yaml'
```
根据这个配置文件，sandbox：

1. 先执行Data-Juicer数据分析功能，计算每条数据的指定指标，比如`configs/demo/process.yaml`中，指定`language_id_score_filter`计算了语言分。

2. 利用Data-Juicer数据分析的结果，用k-sigma方法微调数据菜谱。注意这里设置`meta_name`与数据分析时的`meta_name`相同才能利用到分析结果。

3. 用k-sigma方法微调后的菜谱执行Data-Juicer的数据筛选功能。

4. 用筛选后的数据训练模型。

5. 分析筛选后的数据。

6. 用打分器给筛选后的数据打分。

如果您的配置文件中需要多个 pipeline ，您可以为每个管道命名，并将它们组织在 `pipelines` 字段中：

```yaml
# Sandbox config example

# global parameters
project_name: 'demo-sandbox'
experiment_name: 'demo-sandbox-run0'              # for wandb tracer name
hpo_config: null                                  # path to a configuration file when using auto-HPO tool.

pipelines:
  pipeline_1:
    probe_job_configs:
      xxx
  pipeline_2:
    probe_job_configs:
      xxx
    refine_recipe_job_configs:
      xxx
  pipeline_3:
    probe_job_configs:
      xxx
    execution_job_configs:
      xxx
```

在本例中，`pipelines` 字段包括 3 个 pipeline，分别名为 `pipeline_1`、`pipeline_2` 和 `pipeline_3`。它们各自都有不同类型的作业。您可以在 [这里](https://github.com/datajuicer/data-juicer-sandbox/blob/main/configs/internvl_coco_caption/sandbox_internvl_coco_caption.yaml) 找到 InternVL 沙盒实验的此类配置文件的实际示例。

对于单 pipeline 格式，这个唯一的 pipeline 会默认命名为 "anonymous"。

> [!Important]
> 
> 不包含 `pipelines` 字段的单 pipeline 格式和包含 `pipelines` 字段的多 pipeline 格式均受支持，但不能同时使用。


### 运行沙盒

沙盒的运行入口为`dj-sandbox`，使用方法和数据处理与分析工具类似，需要指定沙盒配置文件：

```yaml
# in data-juicer-sandbox
dj-sandbox --config configs/sandbox/sandbox.yaml
```

运行开始后，沙盒会根据预定义好的流水线以及配置文件依次运行各个步骤。流水线默认的单次运行主要包括4个大步骤：

1. **数据/模型洞察**：该步骤会对输入的原始数据集/模型进行洞察，如对数据集进行分析或者对模型推理产出的数据进行分析，作为后续步骤的指导
2. **基于洞察结果的数据菜谱微调迭代**：该步骤会基于数据/模型的洞察结果，对输入的数据菜谱进行超参微调优化，如根据数据洞察结果可以使用k-sigma方法调整数据菜谱中的算子超参
3. **数据处理与模型训练**：该步骤会基于微调后的数据菜谱对输入的原始数据集进行处理清洗，如沙盒中配置了训练模型步骤，则还会使用处理后的数据集对配置好的模型进行训练
4. **数据/模型评估**：该步骤针对处理后的数据和前一步中训练好的模型（如有）进行评估，评估方法根据配置可包括数据集二次分析，指定benchmark评估等

如此便完成了一轮沙盒流水线运行，最终用户只需比较数据菜谱微调以及数据集处理前后的洞察结果和评估结果，即可验证该轮实验对于数据生产的有效性。

如果在配置文件里设置了`hpo_config`，并在其中配置了合适的优化目标以及待优化的算子超参，则沙盒会以HPO的形式进行多轮的流水线运行，并自动进行算子超参的多轮迭代微调优化。该配置文件的准备可参考 [hpo工具](https://github.com/datajuicer/data-juicer/tree/main/data_juicer/tools/hpo) 。

## 组件工厂

在沙盒流水线的单次运行中，包括了四个大的步骤，其中涉及到如下一些可配置组件，他们分别对应了一个用于初始化这些组件的工厂类：

- **数据处理（DataExecutor）**：数据处理的执行器，即Data-Juicer的Executor
- **数据池操作（DataPoolManipulator）**：数据池的操作，例如构建、组合
- **通用数据处理（GeneralDataExecutor）**：数据集处理的通用执行器，例如数据集格式转换
- **数据分析（DataAnalyzer）**：数据分析器，即Data-Juicer的analyzer
- **数据评估（DataEvaluator）**：数据集质量的评估器
- **通用数据探测（GeneralProbe）**：数据集的通用探测组件
- **模型数据评估（ModelInferEvaluator）**：利用模型推理结果的数据集质量的评估器
- **模型训练（ModelTrainExecutor）**：模型训练执行器
- **模型推理（ModelInferExecutor）**：模型推理执行器
- **模型评估（ModelEvaluator）**：模型性能的评估器

除了DataExecutor和DataAnalyzer，其余组件均可在配置文件中指定`type`参数来选择具体的执行或者评估类型，如数据评估组件支持`type`为`"dj_text_quality_classifier"`来使用Data-Juicer的质量分类器工具来对数据集进行评估，而模型训练组件`type`为`"modelscope"`来训练来自于ModelScope平台的模型。

目前支持的组件工厂以及工厂中支持的组件包括：

- 数据处理工厂 -- DataExecutorFactory

| 组件 | 功能 | `run`方法说明 | 参考材料 |
| --- | --- | --- | --- |
| `DJExecutor` | Data-Juicer数据处理模块 | - | - |

- 数据池操作工厂 -- DataPoolManipulatorFactory

| 组件                      | 功能                 | `run`方法说明              | 参考材料                                               |
|-------------------------|--------------------|------------------------|----------------------------------------------------|
| `DataPoolConstruction`  | 从指定的已分析数据源构建数据池    | -                      | [Sandbox Paper](https://arxiv.org/abs/2407.11784)  |
| `DataPoolCombination`   | 组合指定的数据池           | -                      | [Sandbox Paper](https://arxiv.org/abs/2407.11784)  |
| `DataPoolDuplication`   | 按指定次数复制数据池         | -                      | [Sandbox Paper](https://arxiv.org/abs/2407.11784)  |
| `DataPoolDownsampling`  | 将数据池随机下采样到指定规模     | -                      | [Sandbox Paper](https://arxiv.org/abs/2407.11784)  |
| `DataPoolRanking`       | 根据指定的评测指标对数据池进行排序  | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |
| `DataPoolMerging`       | 将多个数据池合并为一个数据集或数据池 | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |
| `DataPoolCartesianJoin` | 计算两个数据池集合的卡氏积      | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |

- 通用数据处理工厂 -- GeneralDataExecutorFactory

| 组件                            | 功能                                                         | `run`方法说明              | 参考材料                                                                                                     |
|-------------------------------|------------------------------------------------------------|------------------------|----------------------------------------------------------------------------------------------------------|
| `COCOCaptionToDJConversion`   | 将 InternVL COCO Caption 数据集转换为 DJ 格式                       | -                      | [InternVL COCO Caption](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html)  |
| `COCOCaptionMetaGeneration`   | 为 InternVL COCO Caption 数据集生成 meta 文件                      | -                      | [InternVL COCO Caption](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html)  |


- 数据分析工厂 -- DataAnalyzerFactory

| 组件 | 功能 | `run`方法说明 | 参考材料 |
| --- | --- | --- | --- |
| `DJAnalyzer` | Data-Juicer数据分析模块 | - | - |

- 数据评估工厂 -- DataEvaluatorFactory

| 组件 | 功能                                     | `run`方法说明 | 参考材料 |
| --- |----------------------------------------| --- | --- |
| `Gpt3QualityEvaluator` | 使用Data-Juicer复现的GPT-3文本质量分类器对数据集进行质量评估 | <br />- `eval_type`：该评估器评估对象类型，目前只支持`"data"`<br />- `eval_obj`：未使用的参数<br />- 返回值：待评估数据集样本质量打分均值<br /> | [Data-Juicer质量分类器工具集](https://github.com/datajuicer/data-juicer/tree/main/data_juicer/tools/quality_classifier) |
| `VBenchEvaluator` | 使用VBench对基于prompt生成的视频进行多维度的评估         | <br />- `eval_type`：该评估器评估对象类型，目前只支持`"data"`<br />- `eval_obj`：未使用的参数<br />- 返回值：待评生成视频集各维度打分均值<br /> | [VBench论文](https://arxiv.org/abs/2311.17982) |
| `InceptionEvaluator` | 通过视频分类模型抽取特征测评生成的视频                    | <br />- `eval_type`：该评估器评估对象类型，目前只支持`"data"`<br />- `eval_obj`：未使用的参数<br />- 返回值：根据给定的metric返回对应的字典<br /> | [Inception Metrics](https://github.com/NVlabs/long-video-gan/tree/main/metrics) |
| `AccuracyEvaluator` | 评测预测标签和真实标签比较得到的准确率                    | <br />- `eval_type`：该评估器评估对象类型，目前只支持`"data"`<br />- `eval_obj`：未使用的参数<br />- 返回值：根据给定的metric返回对应的字典<br /> | [Inception Metrics](https://github.com/NVlabs/long-video-gan/tree/main/metrics) |
| `MSEEvaluator` | 评测预测值和真实值之间的MSE分数                      | <br />- `eval_type`：该评估器评估对象类型，目前只支持`"data"`<br />- `eval_obj`：未使用的参数<br />- 返回值：根据给定的metric返回对应的字典<br /> | [Inception Metrics](https://github.com/NVlabs/long-video-gan/tree/main/metrics) |

- 通用数据探测工厂 -- GeneralProbeFactory

| 组件                 | 功能                                                                          | `run`方法说明              | 参考材料                                               |
|--------------------|-----------------------------------------------------------------------------|------------------------|----------------------------------------------------|
| `DataPoolRanking`  | 根据指定的评估指标对数据池进行排序                                                           | -                      | [Sandbox Paper](https://arxiv.org/abs/2407.11784)  |


- 模型数据评估工厂 -- ModelInferEvaluatorFactory

| 组件 | 功能 | `run`方法说明 | 参考材料 |
| --- | --- | --- | --- |
| `ModelscopeInferProbeExecutor` | 用数据集对ModelScope平台上的模型进行推理，并返回推理结果 | <br />- `run_type`：推理类型。需要在组件配置文件中设置`type`参数为`"modelscope"`来激活该组件<br />- `run_obj`：需要送入模型推理的采样数据集<br /> | [ModelScope模型推理文档](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8E%A8%E7%90%86Pipeline) |

- 模型训练工厂 -- ModelTrainExecutorFactory

| 组件                                 | 功能                                                                             | `run`方法说明                                                                                           | 参考材料                                                                                                    |
|------------------------------------|--------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `ModelscopeTrainExecutor`          | 用Data-Juicer产出的数据集训练ModelScope平台上的模型，并监测loss变化信息                               | <br />- `run_type`：训练模型类型。需要在组件配置文件中设置`type`参数为`"modelscope"`来激活该组件<br />- `run_obj`：未使用的参数<br />   | [ModelScope模型训练文档](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83Train)       |
| `EasyAnimateTrainExecutor`         | 用Data-Juicer产出的数据集训练文生视频模型EasyAnimate的LoRA模型，并监测loss变化信息                       | <br />- `run_type`：训练模型类型。需要在组件配置文件中设置`type`参数为`"easyanimate"`来激活该组件<br />- `run_obj`：未使用的参数<br />  | [EasyAnimate](https://github.com/aigc-apps/EasyAnimate)                                                 |
| `TrinityRFTTrainExecutor`          | 基于 Trinity-RFT 框架进行强化微调任务，并监测训练状态变化信息                                          | - `run_obj`: 可以是 Trinity 配置文件的路径                                                                    | [Trinity-RFT](https://github.com/modelscope/Trinity-RFT)                                                |
| `InternVLCOCOCaptionTrainExecutor` | 针对 COCO Caption 任务微调 InternVL2 的 LoRA 模型，并监测训练loss和学习率的变化信息                    | -                                                                                                   | [InternVL COCO Caption](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html) |


- 模型推理工厂 -- ModelInferExecutorFactory

| 组件 | 功能                                           | `run`方法说明                                                                                        | 参考材料 |
| --- |----------------------------------------------|--------------------------------------------------------------------------------------------------| --- |
| `EasyAnimateInferExecutor` | 用VBench的prompt数据集对EasyAnimate模型进行推理，并存储生成的视频 | <br />- `run_type`：推理类型。需要在组件配置文件中设置`type`参数为`"easyanimate"`来激活该组件<br />- `run_obj`：未使用的参数<br /> | [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) |
| `HFTransformersInferExecutor` | 用HuggingFace Transformers进行推理。               | <br />- `run_type`：推理类型。需要在组件配置文件中设置`type`参数为`"huggingface"`来激活该组件<br />- `run_obj`：未使用的参数<br /> | -                                                       |
| `VLLMInferExecutor` | 用vLLM进行推理。                                   | <br />- `run_type`：推理类型。需要在组件配置文件中设置`type`参数为`"vllm"`来激活该组件<br />- `run_obj`：未使用的参数<br />        | -                                                       |
| `APIModelInferExecutor` | 用OpenAI API模型进行推理。                           | <br />- `run_type`：推理类型。需要在组件配置文件中设置`type`参数为`"api"`来激活该组件<br />- `run_obj`：未使用的参数<br />         | -                                                       |

- 模型评估工厂 -- ModelEvaluatorFactory

| 组件                              | 功能                                                                       | `run`方法说明              | 参考材料                                                                                                                                     |
|---------------------------------|--------------------------------------------------------------------------|------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `InternVLCOCOCaptionEvaluator`  | 为 InternVL COCO Caption 任务评测 Bleu-1/2/3/4 ，METEOR ， ROUGE_L ， 和 CIDEr 指标 | -                      | [InternVL COCO Caption](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html#evaluating-the-fine-tuned-model)  |


详细定义可参考[组件工厂](https://github.com/datajuicer/data-juicer-sandbox/blob/main/data_juicer_sandbox/factories.py)。

## 上下文共享

有时不同的 hook 之间需要共享信息。例如，`TrainModelHook` 需要与 `EvaluateModelHook` 共享模型检查点路径，以便在训练后对模型进行评估。

为了实现这一点，我们集成了一个全局的、跨 hook 和跨 pipeline 的信息容器，名为 `context_infos`，用于存储每个 pipeline 中 hook 的执行结果。`context_infos` 会在管道执行时自动构建。

### Resume 模式

每个 pipeline 完成后，`context_infos` 都会保存在磁盘上。基于此，Sandbox 允许 pipeline 级别的 Resume 模式，从上一个 pipeline 的检查点继续执行沙盒。我们只需在沙盒配置文件中将 `resume` 参数设置为 `true` 即可。

### Hooks 的输入、输出和本地参数

为了获取上下文信息，Hooks 新增了三个参数：
- `input`：用于获取存储在上下文信息中之前的 Hook 的执行结果，并将其更新到当前 hook 的配置中。
  - 基本用法：`<key_to_updated>: <key_in_context_infos>`，其中 `<key_to_updated>` 是当前 Hook 配置中需要更新的键，`<key_in_context_infos>` 是存储在上下文信息中的所需结果的键。
  - 每对 input 参数都会将 `<key_to_updated>` 中的值替换为上下文信息中键为 `<key_in_context_infos>` 的值。它们都支持使用点 (`.`) 操作符的嵌套键。
  - 如果只需要上一个 hook 的结果，可以在 `<key_in_context_infos>` 中简单地使用 `-1` 作为 hook 的键。
- `output`：用于将当前 hook 的结果存储在 `context_infos ` 中。
  - 基本用法：`[<res_1_name>, <res_2_name>, ...]`，其中 `<res_i_name>` 表示当前 hook 的第 `i` 个输出结果。如果未指定 `output` 参数，则自动使用简单的 "res_i" 作为第 `i` 个输出结果的名称。
  - hook 执行后，当前 hook 的结果将以字典形式存储在 `context_infos ` 中，并使用 `output` 参数中的名称作为键。
- `local`：用于将指定的值更新到当前 hook 的配置中。
  - 基本用法：`<key_to_updated>: <value>`，其中 `<key_to_updated>` 是当前钩子配置中待更新的键，`<value>` 是目标值。
  - 每对 local 参数都会将 `<key_to_updated>` 中的值替换为 `<value>` 中指定的值。支持使用点 (`.`) 符号的嵌套键。

使用这些参数的一个钩子示例如下：

```yaml
xxx_hook:
  meta_name: job_xxx
  input:
    dj_configs.dataset_path: pipeline1.name3.res4_key
    extra_configs.meta_paths: -1.res5_key
  output: ['res6_key', 'res7_key']
  local:
    extra_configs.arg2: 42
  dj_configs: '/path/to/dj_configs.yaml'
  extra_configs:
    arg1: "arg1_val"
    arg2: 404
    meta_paths: "<placeholder>"
```

此钩子使用了所有的三个参数：
- 在 `input` 中，此钩子将 YAML 格式的 `dj_configs` 中的 `dataset_path` 替换为之前钩子上下文信息中 pipeline1 中 `meta_name` 为 "name3" 的钩子存储名称为 `res4_key` 的值。此外，它会将 `extra_configs` 中的 `meta_paths` 替换为上一个钩子（由"-1"指定）上下文信息中存储的 `res5_key` 的值。
- 在 `output` 中，该钩子输出两个结果，分别命名为 `res6_key` 和 `res7_key`，它们将存储在上下文信息中，如下所示：
```python
{
  'meta_name': 'job_xxx',
  'res6_key': <output_1>,
  'res7_key': <output_2>,
}
```
- 在 `local` 中，该钩子将 `extra_configs` 中 `arg2` 的原始值（为 404）替换为目标值 42。
