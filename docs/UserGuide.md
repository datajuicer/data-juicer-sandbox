# User Guide


## What is DJ-Sandbox?

In Data-Juicer, the DJ-Sandbox is a middleware that links data and model feedback, enabling high performance and low-cost verification across a wide range of tasks. It aims to provide users with the best practices for continuously enhancing data-model recipes, featuring low overhead, portability, and guidance. In the sandbox, users can quickly experiment, iterate, and refine recipes based on small-scale datasets and models before scaling up to produce high-quality data to serve large-scale models.

In addition to the basic data optimization and recipe refinement features offered by Data-Juicer, users can seamlessly use configurable components such as data probing and analysis, model training and evaluation, and data and model feedback-based recipe refinement to form preferred pipelines for data-model research and development.

For more detailed information, please refer to our [paper](http://arxiv.org/abs/2407.11784) (ICML'25 spotlight).

## Applications and Use-Cases
We apply the sandbox to many cutting-edge models, such as Mini-Gemini and InternVL-2.0 (two LLaVA-inspired models for image-to-text generation), EasyAnimate and T2V-Turbo (two Diffusion Transformer-based models for text-to-video generation), and a CLIP model for image-text pre-training. Among these, we have secured a new leading position on the [VBench](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard) text-to-video leaderboard.
![top-1_in_vbench](https://img.alicdn.com/imgextra/i1/O1CN01I9wHW91UNnX9wtCWu_!!6000000002506-2-tps-1275-668.png)

The model is now publicly available on the ModelScope and HuggingFace platforms, and the training dataset has also been available.

| Open-source model or dataset | Link | Description |
| ------------ | --- | --- |
| Data-Juicer (T2V, 147k) |  [ModelScope](https://modelscope.cn/models/Data-Juicer/Data-Juicer-T2V) <br> [HuggingFace](https://huggingface.co/datajuicer/Data-Juicer-T2V) | Corresponding to Data-Juicer (T2V-Turbo) model in VBench leaderboard |
| Data-Juicer (DJ, 228k) | [ModelScope](https://modelscope.cn/models/Data-Juicer/Data-Juicer-T2V-v2) <br> [HuggingFace](https://huggingface.co/datajuicer/Data-Juicer-T2V-v2) | Corresponding to Data-Juicer (2024-09-23, T2V-Turbo) model in VBench leaderboard |
| data_juicer_t2v_optimal_data_pool | [Aliyun](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/data_juicer_t2v_optimal_data_pool.zip) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/data-juicer-t2v-optimal-data-pool)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/data-juicer-t2v-optimal-data-pool) | The training dataset of Data-Juicer (T2V, 147k) |
| data_juicer_t2v_evolution_data_pool | [Aliyun](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/data_juicer_t2v_optimal_data_pool_s2.zip) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/data-juicer-t2v-evolution-data-pool) | The training dataset of Data-Juicer (2024-09-23, T2V-Turbo) |

Following is the case study for Data-Juicer (DJ, 228k) outputs.
  | Prompt | Generated Video |
  | --- | --- |
  | A beautiful coastal beach in spring, waves lapping on sand, zoom out | [![Case 0](https://img.alicdn.com/imgextra/i1/O1CN01KuJeOE1Ylqnk9zYkc_!!6000000003100-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case0.mp4) |
  | a boat accelerating to gain speed | [![Case 1](https://img.alicdn.com/imgextra/i2/O1CN01i1iMFE1TKlIUlqE8d_!!6000000002364-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case1.mp4) |
  | A boat sailing leisurely along the Seine River with the Eiffel Tower in background by Hokusai, in the style of Ukiyo | [![Case 2](https://img.alicdn.com/imgextra/i2/O1CN01u2cjJE1RBwRFeCFuo_!!6000000002074-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case2.mp4) |
  | a bottle on the left of a wine glass, front view | [![Case 3](https://img.alicdn.com/imgextra/i4/O1CN01vdMm6Q1xWc1CoJZW6_!!6000000006451-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case3.mp4) |
  | A corgi's head depicted as an explosion of a nebula | [![Case 4](https://img.alicdn.com/imgextra/i2/O1CN014oPB8Q1IrJg0AbUUg_!!6000000000946-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case4.mp4) |
  | A graceful ballerina doing a pirouette on a dimly lit stage, with soft spotlight highlighting her movements. | [![Case 5](https://img.alicdn.com/imgextra/i4/O1CN01yNlsVu1ymvkJgkvY8_!!6000000006622-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case5.mp4) |

To reproduce the paper's experiments, please refer to the sandbox usage guide below, the experimental process in the following figure, the [initial dataset](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/data_juicer_t2v_init_data_pool.zip), and the configuration file demos for the process: [1_single_op_pipeline.yaml](https://github.com/datajuicer/data-juicer-sandbox/tree/main/configs/easyanimate_text_to_video/1_single_op_pipeline.yaml), [2_multi_op_pipeline.yaml](https://github.com/datajuicer/data-juicer-sandbox/tree/main/configs/easyanimate_text_to_video/2_multi_op_pipeline.yaml), [3_duplicate_pipeline.yaml](https://github.com/datajuicer/data-juicer-sandbox/tree/main/configs/easyanimate_text_to_video/3_duplicate_pipeline.yaml).
![bench_bottom_up](https://img.alicdn.com/imgextra/i2/O1CN01xvu2fo1HU80biR6Q5_!!6000000000760-2-tps-7756-3693.png)


## Quick Start

### Requirements

Before using sandbox, you need to install data-juicer-sandbox by running the command below:
```shell
git clone https://github.com/datajuicer/data-juicer-sandbox.git
cd data-juicer-sandbox/
uv pip install -e ".[all]"
```
And prepare third-party libraries used in sandbox (e.g., EasyAnimate, VBench, InternVL, etc.) according to their official instructions, or you can simply clone the third-party repositories from GitHub and leave the installation process to our `EnvManager` during sandbox running.

**NOTICE**: some sandbox-related dependencies require extra domain dependencies. 

1. To use [ModelScope](https://github.com/modelscope/modelscope), you need to install the related dependencies from its independent host:
```shell
uv pip install "modelscope[framework,nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
Please refer to the [ModelScope documentation](https://www.modelscope.cn/docs/Beginner-s-Guide/Environment-Setup) for more information.


2. To use [EasyAnimate](https://github.com/aigc-apps/EasyAnimate), you need to execute the following installation script:
```shell
cd thirdparty/models/
bash setup_easyanimate.sh
cd ../../
```

If some Module-Not-Found errors are raised by these third-party libraries when running the sandbox, users need to check their docs first.

### Prepare Configuration Files for Sandbox

The sandbox will sequentially execute four types of jobs: Data/Model Probe (`probe_job_configs`), Iterative Recipe Refinement based on Probe Results(`refine_recipe_job_configs`), Dataset Processing and Model Training (`execution_job_configs`) and Data/Model Evaluation (`evaluation_job_configs`). Within each category of jobs, jobs are carried out in the order specified by the configured job list. Each task requires specifying: the hook for mounting this job (`hook`), the tag name for identifying the hook (`meta_name`), Data-Juicer data processing parameters (`dj_configs`), as well as other specific parameters for the job (`extra_configs`). Among these parameters, hook is required, while others may be left empty. dj_configs can refer to the full Data-Juicer data processing parameters available in [config_all.yaml](https://github.com/datajuicer/data-juicer/blob/main/data_juicer/config/config_all.yaml). The `extra_configs` are task-specific parameters without restrictions. They can include parameters for model training, inference, evaluation, etc. For example, `path_k_sigma_recipe` can be used to specify the path for saving the data recipe refined using the k-sigma method. An example of a sandbox configuration file can be found at `configs/demo/sandbox/sandbox.yaml`:

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
Based on this configuration file, sandbox:

1. Execute the Data-Juicer data analysis function to calculate specified metrics for each piece of data, for example, in `configs/demo/process.yaml`, the `language_id_score_filter` is designated to calculate language scores.

2. With the results from Data-Juicer data analysis, fine-tune the data recipe using the k-sigma method. Note that the `meta_name` here is set the same as the `meta_name` used during data analysis to query the results from W&B.

3. Execute Data-Juicer's data filtering function with the data recipe fine-tuned by the k-sigma method.

4. Train the model with the filtered data.

5. Analyze the data after filtering.

6. Score the data after filtering with a scorer.

When there are multiple pipelines needed in your config file, you can name each pipeline and organize them in a `pipelines` field:

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

In this example, there are 3 pipelines organized in the `pipelines` field, named `pipeline_1`, `pipeline_2`, and `pipeline_3`. Each of them has their own different types of jobs. You can find a practical example of such config file for InternVL sandbox experiments [here](https://github.com/datajuicer/data-juicer-sandbox/blob/main/configs/internvl_coco_caption/sandbox_internvl_coco_caption.yaml).

For the single-pipeline format, the only pipeline is named "anonymous" in default.

> [!Important]
> 
> The single pipeline format without `pipelines` field and the multi-pipeline format with `pipelines` field are both supported but cannot be used at the same time.

### Start Sandbox

The entry point for running the sandbox is `dj-sandbox`. The usage is similar to the data processing and analysis tool, requiring specifying the sandbox configuration file:

```yaml
# in data-juicer-sandbox
dj-sandbox --config configs/demo/sandbox.yaml
```

Once the run is started, the sandbox will sequentially execute each of the predefined pipeline steps according to the configuration file. The default one trial of the pipeline mainly includes four major steps:

1. **Data/Model Probe**: This step provides probes into the input dataset/model, such as analysing the dataset or analysing the data produced by model inference, to guide the subsequent steps.
2. **Iterative Recipe Refinement based on Probe Results**: This step refines and optimizes the recipe hyperparameters based on the data/model probes. For example, the operator (OP) hyperparameters in the data recipe can be adjusted using the k-sigma method based on the data probes.
3. **Dataset Processing and Model Training**: This step processes and cleans the input dataset based on the refined recipe. If model training is configured in the sandbox, the processed dataset will also be used to train the configured model.
4. **Data/Model Evaluation**: This step evaluates the processed dataset and the model trained in the previous step (if applicable). The evaluation methods may include analysis of the processed dataset and specified benchmark evaluations based on the configuration.

Once this completes one trial of the sandbox pipeline run, the user can validate the effectiveness of the experiment in data production by comparing the probes and evaluation results before and after recipe refinement and dataset processing.

If the `hpo_config` is set in the configuration file and appropriate optimization objectives and OP hyperparameters to be refined are configured within it, the sandbox will perform multiple trials of pipeline runs in the form of Hyperparameter Optimization (HPO) and automatically conduct iterative refinement and optimization of the operator hyperparameters. The preparation of this configuration file can be referenced from the [HPO tool](https://github.com/datajuicer/data-juicer/tree/main/data_juicer/tools/hpo).

## Component Factory

In a single trial of the sandbox pipeline, four major steps involve various configurable components. Each of these components corresponds to a factory class used to initialize them:

- **Data Processing (DataExecutor)**: Executor for dataset processing, i.e., the Executor of Data-Juicer
- **Data Pool Manipulator (DataPoolManipulator)**: Manipulator for data pools, i.e., construction, combination
- **General Data Processing (GeneralDataExecutor)**: General executor for dataset processing, i.e., dataset format conversion
- **Data Analyzing（DataAnalyzer）**: Analyzer for dataset, i.e., the analyzer of Data-Juicer
- **Data Evaluation (DataEvaluator)**: Evaluator on the quality of the dataset
- **General Data Probe (GeneralProbe)**: General probe components for the dataset
- **Model-Data Evaluation (ModelInferEvaluator)**: Evaluator of dataset quality using the model's inference results
- **Model Training (ModelTrainExecutor)**: Executor for model training
- **Model Inference (ModelInferExecutor)**: Executor for model inference
- **Model Evaluation (ModelEvaluator)**: Evaluator on the performance of the model

Except for DataExecutor and DataAnalyzer, the rest of the components can be specified in the configuration file using the `type` parameter to choose a specific execution or evaluation type. For example, the data evaluation component supports a `type` of `"dj_text_quality_classifier"` to utilize Data-Juicer's text quality classifier tool for evaluating the dataset, while the model training component `type` can be set to `"modelscope"` to train a model from the ModelScope platform.

The currently supported component factories and the components supported within each factory are as follows:

- DataExecutorFactory

| Component | Function | Desc. of Method `run` | Reference Materials |
| --- | --- | --- | --- |
| `DJExecutor` | The data process module of Data-Juicer | - | - |

- DataPoolManipulatorFactory

| Component                | Function                                                  | Desc. of Method `run` | Reference Materials                               |
|--------------------------|-----------------------------------------------------------|-----------------------|---------------------------------------------------|
| `DataPoolConstruction`   | Construct data pool from specified analyzed data source   | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |
| `DataPoolCombination`    | Combine specified data pools                              | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |
| `DataPoolDuplication`    | Duplicate a data pool for specified times                 | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |
| `DataPoolDownsampling`   | Randomly downsample data pools to specified scale         | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |
| `DataPoolRanking`        | Rank data pools according to specified evaluation metrics | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |
| `DataPoolMerging`        | Merge data pools into one dataset or data pool            | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |
| `DataPoolCartesianJoin`  | Join two sets of data pools with Cartesian Join           | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |


- GeneralDataExecutorFactory

| Component                   | Function                                                     | Desc. of Method `run` | Reference Materials                                                                                     |
|-----------------------------|--------------------------------------------------------------|-----------------------|---------------------------------------------------------------------------------------------------------|
| `COCOCaptionToDJConversion` | Convert InternVL COCO Caption datasets to DJ format          | -                     | [InternVL COCO Caption](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html) |
| `COCOCaptionMetaGeneration` | Generate meta file for InternVL COCO Caption datasets        | -                     | [InternVL COCO Caption](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html) |

- DataAnalyzerFactory

| Component | Function | Desc. of Method `run` | Reference Materials |
| --- | --- | --- | --- |
| `DJAnalyzer` | The data analysis module of Data-Juicer | - | - |

- DataEvaluatorFactory

| Component | Function                                                                                             | Desc. of Method `run` | Reference Materials |
| --- |------------------------------------------------------------------------------------------------------| --- | --- |
| `Gpt3QualityEvaluator` | Evaluate the quality of a dataset using the GPT-3 text quality classifier reproduced by Data-Juicer. | <br />- `eval_type`: The type of the object to be evaluated by the evaluator, currently only supports `"data"`.<br />- `eval_obj`: A useless parameter.<br /> | [Data-Juicer Quality Classifier Toolkit](https://github.com/datajuicer/data-juicer/tree/main/data_juicer/tools/quality_classifier) |
| `VBenchEvaluator` | Evaluate the generated videos according to given prompts in multi dimensions                         | <br />- `eval_type`: The type of the object to be evaluated by the evaluator, currently only supports `"data"`<br />- `eval_obj`: A useless parameter.<br />- Return: The average score of generated videos in multi dimensions.<br /> | [VBench paper](https://arxiv.org/abs/2311.17982) |
| `InceptionEvaluator` | Evaluate the generated videos by features extracted from video classification models.                | <br />- `eval_type`: The type of the object to be evaluated by the evaluator, currently only supports `"data"`<br />- `eval_obj`: A useless parameter.<br />- Return: A dictionary of scores in the given metric. <br /> | [Inception Metrics](https://github.com/NVlabs/long-video-gan/tree/main/metrics) |
| `AccuracyEvaluator` | Evaluate the accuracy to compare the labels in the predicted ones and ground truth                   | <br />- `eval_type`: The type of the object to be evaluated by the evaluator, currently only supports `"data"`<br />- `eval_obj`: A useless parameter.<br />- Return: A dictionary of scores in the given metric. <br /> | [Inception Metrics](https://github.com/NVlabs/long-video-gan/tree/main/metrics) |
| `MSEEvaluator` | Evaluate the MSE score between the predicted values and ground truth.                | <br />- `eval_type`: The type of the object to be evaluated by the evaluator, currently only supports `"data"`<br />- `eval_obj`: A useless parameter.<br />- Return: A dictionary of scores in the given metric. <br /> | [Inception Metrics](https://github.com/NVlabs/long-video-gan/tree/main/metrics) |

- GeneralProbeFactory

| Component         | Function                                                    | Desc. of Method `run` | Reference Materials                               |
|-------------------|-------------------------------------------------------------|-----------------------|---------------------------------------------------|
| `DataPoolRanking` | Rank data pools according to specified evaluation metrics   | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |

- ModelInferEvaluatorFactory

| Component | Function | Desc. of Method `run` | Reference Materials |
| --- | --- | --- | --- |
| `ModelscopeInferExecutor` | Perform inference on a model from the ModelScope platform using a specified sampled dataset, and return the inference results. | <br />- `run_type`: Type of model inference. We need to set `type`  arg as `"modelscope"` in the component configuration file to activate this component.<br />- `run_obj`: Sampled dataset to be fed into model inference.<br /> | [ModelScope Docs of Model Inference](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8E%A8%E7%90%86Pipeline) |

- ModelTrainExecutorFactory

| Component                          | Function                                                                                                                           | Desc. of Method `run`                                                                                                                                                                                   | Reference Materials                                                                                                |
|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `ModelscopeTrainExecutor`          | Perform a training task on a model from the ModelScope platform using specified datasets, and monitor the change in training loss. | <br />- `run_type`: Type of model training. We need to set `type`  arg as `"modelscope"` in the component configuration file to activate this component.<br />- `run_obj`: A useless parameter.<br />  | [ModelScope Docs of Model Training](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83Train) |
| `EasyAnimateTrainExecutor`         | Perform a LoRA training task on EasyAnimate text-to-video model, and monitor the change in training loss.                          | <br />- `run_type`: Type of model training. We need to set `type`  arg as `"easyanimate"` in the component configuration file to activate this component.<br />- `run_obj`: A useless parameter.<br /> | [EasyAnimate](https://github.com/aigc-apps/EasyAnimate)                                                            |
| `TrinityRFTTrainExecutor`          | Perform a reinforcement fine-tuning task based on Trinity-RFT framework, and monitor the change in training states.                | - `run_obj`: Could be the path to Trinity configs.                                                                                                                                                      | [Trinity-RFT](https://github.com/modelscope/Trinity-RFT)                                                           |
| `InternVLCOCOCaptionTrainExecutor` | Perform a LoRA fine-tuning task on InternVL2 for COCO Caption task, and monitor the change in training loss and learning rate.     | -                                                                                                                                                                                                       | [InternVL COCO Caption](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html)            |


- ModelInferExecutorFactory

| Component | Function                                                                                                          | Desc. of Method `run` | Reference Materials                                     |
| --- |-------------------------------------------------------------------------------------------------------------------| --- |---------------------------------------------------------|
| `EasyAnimateInferExecutor` | Perform inference on EasyAnimate text-to-video model with the prompts from VBench, and save the generated videos. | <br />- `run_type`: Type of model inference. We need to set `type`  arg as `"easyanimate"` in the component configuration file to activate this component.<br />- `run_obj`: A useless parameter.<br /> | [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) |
| `HFTransformersInferExecutor` | Perform inference with HuggingFace Transformers.                                                                  | <br />- `run_type`: Type of model inference. We need to set `type`  arg as `"huggingface"` in the component configuration file to activate this component.<br />- `run_obj`: A useless parameter.<br /> | -                                                       |
| `VLLMInferExecutor` | Perform inference with vLLM.                                                                                      | <br />- `run_type`: Type of model inference. We need to set `type`  arg as `"vllm"` in the component configuration file to activate this component.<br />- `run_obj`: A useless parameter.<br /> | -                                                       |
| `APIModelInferExecutor` | Perform inference with OpenAI API.                                                                                | <br />- `run_type`: Type of model inference. We need to set `type`  arg as `"api"` in the component configuration file to activate this component.<br />- `run_obj`: A useless parameter.<br /> | -                                                       |

- ModelEvaluatorFactory

| Component                      | Function                                                                         | Desc. of Method `run` | Reference Materials                                                                                                                     |
|--------------------------------|----------------------------------------------------------------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `InternVLCOCOCaptionEvaluator` | Evaluate Bleu-1/2/3/4, METEOR, ROUGE_L, and CIDEr for InternVL COCO Caption task | -                     | [InternVL COCO Caption](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html#evaluating-the-fine-tuned-model) |

Please refer to the [component factory](https://github.com/datajuicer/data-juicer-sandbox/blob/main/data_juicer_sandbox/factories.py) for detailed definitions.

## Context Sharing

Sometimes, information needs to be shared between different hooks. For example, the `TrainModelHook` needs to share the model checkpoint path with the `EvaluateModelHook` to perform evaluation on the model after training.

To achieve this, we integrate a global, both cross-hook and cross-pipeline information container called `context_infos` to store the executed results of hooks in each pipeline. The `context_infos` are constructed automatically when the pipeline is executed.

### Resume Mode

After each pipeline is finished, the `context_infos` are saved on disk. Based on this, Sandbox allows a pipeline-level resume mode to continue the sandbox execution from the last pipeline checkpoint. All we need to do is to set the `resume` parameter in the sandbox configuration file to `true`.

### Input, Output, and Local Parameter of Hooks

To obtain the context infos, three new parameters are added to the hook:
- `input`: used to obtain results from the previous hook stored in the context infos, and update it in the configuration of the current hook.
  - Basic usage: `<key_to_updated>: <key_in_context_infos>`, where `<key_to_updated>` is the key in the configuration of the current hook to be updated, and `<key_in_context_infos>` is the key of the previous result stored in the context infos.
  - Each pair of input parameters would replace the values in the `<key_to_updated>` with the values in the context infos with key `<key_in_context_infos>`. Nested keys using dot (`.`) notation are supported for both of them.
  - If only the result from the last hook is needed, a simple `-1` can be used as the hook key in the `<key_in_context_infos>`.
- `output`: used to store the results of the current hook in the context infos.
  - Basic usage: `[<res_1_name>, <res_2_name>, ...]`, where `<res_i_name>` represents the `i`th output result of the current hook. If the `output` parameter is not specified, simple "res_i" is used automatically for the `i`th output result.
  - After the hook is executed, the results of the current hook are stored as dict in the context infos with the keys specified in the `output` parameter.
- `local`: used to update the configuration of the current hook locally with specified values.
  - Basic usage: `<key_to_updated>: <value>`, where `<key_to_updated>` is the key in the configuration of the current hook to be updated, and `<value>` is the target value.
  - Each pair of local parameters would replace the values in the `<key_to_updated>` with the values specified in the `<value>`. Nested keys using dot (`.`) notation are supported.

An example of a hook using these parameters is shown below:

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

In this hook, it uses all three parameters:
- In `input`, the hook replaces the `dataset_path` in the `dj_configs` in YAML format with the value of the `res4_key` stored in the context infos of the previous hook with `meta_name` "name3" in the pipeline named "pipeline1". Besides, it replaces the `meta_paths` in the `extra_configs` with the value of the `res5_key` stored in the context infos of the previous hook specified by "-1".
- In `output`, the hook outputs two results named `res6_key` and `res7_key`, which will be stored in the context infos as following:
```python
{
  'meta_name': 'job_xxx',
  'res6_key': <output_1>,
  'res7_key': <output_2>,
}
```
- In `local`, the hook replaces the original value of `arg2` in the `extra_configs`, which is 404 before, with the target value 42.
