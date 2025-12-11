# Developer Guide

As mentioned in the previous section, developers can develop customized configurable components and add them to the corresponding factory classes, then route to appropriate instantiation methods using the `type` parameter. Once the components are implemented, developers can encapsulate them as hooks and register the hooks into the job list. After the job list is orchestrated in the pipeline, when the sandbox pipeline is executed, each job in the job list will be executed in sequence at each step. Each of these parts - components, component factory, hooks, job lists, and the registration and execution orchestration of the pipeline - can be customized by the developer. The relationship among these parts is illustrated in the diagram below.
![sandbox-pipeline](https://img.alicdn.com/imgextra/i3/O1CN01ERmGre1uz3luKOn4n_!!6000000006107-2-tps-4655-1918.png)

## The Internal Implementation of Components
Currently, components are mainly divided into three major categories:

- **Executor**: Since the data executor is already handled by the Data-Juicer's Executor, the executor here specifically refers to the model executor, including model training, inference, evaluation, etc. The code is located [here](https://github.com/datajuicer/data-juicer-sandbox/blob/main/data_juicer_sandbox/model_executors.py).
- **Evaluator**: Used for evaluating the quality and performance of datasets or models. The code is located [here](https://github.com/datajuicer/data-juicer-sandbox/blob/main/data_juicer_sandbox/evaluators.py).
- **DataPoolManipulator**: Used for manipulating the data pool, such as construction, combination, sampling, etc. The code is located [here](https://github.com/datajuicer/data-juicer-sandbox/blob/main/data_juicer_sandbox/data_pool_manipulators.py).

### Executor
The core function of the model executor is to train, infer, or evaluate the model specified in the configuration file with the specified dataset. The model executor needs to inherit from `BaseModelExecutor` and implement several core methods:

- The specific behavior of the model executor (training, inference, evaluation, etc.) needs to be defined in the `_run` method.
- The model executor does not return any value. Key metrics that need to be monitored during execution are usually parsed from the logs produced by the model executor (such as loss, evaluation results, etc.). The parsing and monitoring process needs to be defined by the `_watch_run` method.
- Model executor requires input from a dataset, so the `data_connector` method needs to be implemented to convert the dataset from the sandbox's format to the format required by the model framework or library.

It is important to note that, to monitor the change of training metrics (e.g., loss) promptly during the model training process, logs generated during training need to be monitored. Therefore, both the `_run` method for executing model training and the `watch_run` method for monitoring logs need to be asynchronous methods, indicated by the `async` keyword. In the `run` method, we redirect the standard output stream (stdout) and standard error stream (stderr) to a designated log file before the training starts. Then, we create two asynchronous tasks to execute the aforementioned two methods, each performing the following tasks:

- `_run` method: After loading the dataset, it starts model training based on the model training configuration. Upon completion of training, it outputs a predefined task completion identifier to the standard output stream, which has been redirected to the designated log file.
- `watch_run` method: It monitors the designated log file, reads it line by line, and calls the `_watch_run` method. The called method is customized based on the model training framework and used to parse the latest log content line, extract key metrics, and monitor them until the predefined task completion identifier is read.

### Evaluator

The core function of the evaluator is to evaluate the quality and performance of the target using some specific methods and return the evaluation result, usually a numerical value. The evaluator needs to inherit from the base class `BaseEvaluator` and implement the `run` method. The `run` method typically takes two required parameters:

- `eval_type`: The type of evaluation, used for internal evaluation type routine within a certain evaluator.
- `eval_obj`: The object to be evaluated.

Users can also extend the usage of these two parameters based on their implementation.

### DataPoolManipulator

The core function of the data pool manipulator is to manipulate the data pool, such as construction, combination, sampling, etc. The data pool manipulator needs to inherit from the base class `BaseDataPoolManipulator` and implement the `run` method. The necessary parameters usually come from the input data pool configs in the `__init__` method, covering input data pools, export paths, and specific parameters for each type of manipulators.

Users can refer to the doc string of the `run` method of each manipulator for more details [here](https://github.com/datajuicer/data-juicer-sandbox/blob/main/data_juicer_sandbox/data_pool_manipulators.py).

## Pipeline Hook

As mentioned at the start of this section, in the pipeline, we need to implement several hooks to connect components with the pipeline execution steps through the job list. Activated hooks will be registered in the pipeline's job list and then executed one by one during the pipeline execution at each step. The job lists for the four corresponding steps are as follows:

1. **Data/Model Probe**: Probe job list -- probe_jobs
2. **Iterative Recipe Refinement based on Probe Results**: Refinement job list -- refine_recipe_jobs
3. **Data Processing and Model Training**: Execution job list - execution_jobs
4. **Data/Model Evaluation**: Evaluation job list - evaluation_jobs

In general, we only need to implement one type of hook function for a type of component factory. In addition to hooks that depend on components, some hooks depend on the existing functionality and tools of Data-Juicer or other third-party libraries. The correspondence among these hooks, dependent components, tools, and job lists is as follows:

| Hook | Function                                                                                                      | Dependent Component Factory                          | Dependent Tool or Library                                                                                                                                               | Registered Job List |
| --- |---------------------------------------------------------------------------------------------------------------|------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- |
| `ProbeViaAnalyzerHook` | Analyze and probe the quality and diversity distribution of the dataset                                       | DataAnalyzerFactory                                  | Data-Juicer Analyzer                                                                                                                                                    | - probe_jobs<br />- evaluation_jobs |
| `ProbeViaModelInferHook` | Analyze and understand the impact of the dataset on the model, explore and probe "difficult" and "dirty" data | DataExecutorFactory <br />ModelInferEvaluatorFactory | Data-Juicer Executor                                                                                                                                                    | - probe_jobs<br />- evaluation_jobs |
| `GeneralProbeHook` | General hook for probing the dataset, including ranking the datasets and so on                                | GeneralProbeFactory                                  | -                                                                                                                                                                       | - probe_jobs |
| `RefineRecipeViaKSigmaHook` | Refine data recipe hyperparameters using the k-sigma method based on the probe results of the dataset         | -                                                    | k-sigma recipe refinement tool of Data-Juicer Hyperparameter Optimization (HPO) toolkit                                                                                 | - refine_recipe_jobs |
| `RefineRecipeViaModelFeedbackHook` | Refine data recipe hyperparameters using model probe and feedback results                                     | TODO                                                 | -                                                                                                                                                                       | - refine_recipe_jobs |
| `ProcessDataHook` | Process and clean the dataset based on the current data recipe                                                | DataExecutorFactory                                  | Data-Juicer Executor                                                                                                                                                    | - execution_jobs |
| `DataPoolManipulationHook` | Data pool manipulation,  including construction, combination, sampling, etc.                                  | DataPoolManipulatorFactory                           | -                                                                                                                                                                       | - execution_jobs |
| `GeneralDataExecutorHook` | General data processing for dataset, including format conversion, etc.                                        | GeneralDataExecutorFactory                           | -                                                                                                                                                                       | - execution_jobs |
| `TrainModelHook` | Train a model based on the current dataset                                                                    | ModelTrainExecutorFactory                            | [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) <br/> [InternVL](https://internvl.readthedocs.io/en/latest/index.html)                                                                                                 | - execution_jobs |
| `InferModelHook` | The model generates output based on the given input                                                           | ModelInferExecutorFactory                            | [EasyAnimate](https://github.com/aigc-apps/EasyAnimate)                                                                                                                    | - execution_jobs |
| `EvaluateDataHook` | Evaluate the dataset in terms of data quality and other dimensions                                            | DataEvaluatorFactory                                 | [inception metrics](../tools/mm_eval/inception_metrics/README.md) for images and videos, such as FID and FVD <br /> [VBench](../tools/mm_eval/vbench_metrics/README.md) | - evaluation_jobs |
| `EvaluateModelHook` | Evaluate the trained model                                                                                    | ModelEvaluatorFactory                                | [InternVL COCO Caption](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html#evaluating-the-fine-tuned-model)                                                                                                                                                                       | - evaluation_jobs |

It is worth noting that a hook can be registered in multiple job lists, as this hook can play different roles in different steps of the pipeline. For example, we can analyze and probe both the pre-processed and post-processed datasets to compare the changes in quality, diversity, and other dimensions before and after data processing.

## Customized Sandbox Pipeline
Users can directly modify the job configuration list in the parameter configuration file to achieve task modification and orchestration.

## Watcher
In the above sections, the concept of "monitoring" is repeatedly mentioned. The pipeline will monitor several metrics produced in each step, and these monitoring processes are implemented by `SandboxWatcher`.

`SandboxWatcher` is based on wandb library and mainly includes four methods:

- `setup_sweep`: This method is called in the multi-trial HPO mode, which is supported by the sweep module in wandb library. Therefore, the additional `hpo_config` configuration for sweep initialization is required to be passed into the sandbox configuration file.
- `watch_cfgs`: This method monitors and updates the sandbox experiments and configuration files of various components.
- `watch`: This method monitors a specific metric or experiment result and records it in the wandb log.
- `query`: This method queries a specific metric or experiment result from the wandb log.

## Details of Context Infos

The `context_infos` consists of two levels:

- pipeline level: it's the 1st level of `context_infos`, which is a dict with keys are the pipeline names and values are a list of context information of each job in this pipeline.
- job level: it's the 2nd level of `context_infos`, which is a list of dicts, each dict represents the context information of a specific job, with `meta_name` to identify the job and other key-value pairs with keys are the name of the outputs of this job and values are the output values.

Here is an example of `context_infos`:

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

## Environment Manager

Sandbox supports different kinds of third-party libraries for training, evaluation and so on. If we put all of them into
one environment, version conflicts on some important and complex dependencies will occur. Therefore, we provide an 
easy-to-use environment manager to manage different environments for different third-party libraries, allow users to run
commands in isolated environments independently.

The basic class of environment is `Env` [here](https://github.com/datajuicer/data-juicer-sandbox/blob/main/data_juicer_sandbox/env_manager.py) implemented as below:
```python
class Env(ABC):
  
    @abstractmethod
    def create(self):
        """
        Create an environment.
        """
        raise NotImplementedError(
            'This method must be implemented in subclass.')

    @abstractmethod
    def check_availability(self):
        """
        Check the availability of the environment manager.
        """
        raise NotImplementedError(
            'This method must be implemented in subclass.')

    @abstractmethod
    def exists(self):
        """
        Check if an environment exists.
        """
        raise NotImplementedError(
            'This method must be implemented in subclass.')

    @abstractmethod
    def install_py_deps(self):
        """
        Install Python dependencies.
        """
        raise NotImplementedError(
            'This method must be implemented in subclass.')

    @abstractmethod
    def run_cmd(self):
        """
        Run a command in this environment.
        """
        raise NotImplementedError(
            'This method must be implemented in subclass.')
```

It consists of five main abstract methods:
- `create`: create an environment if it does not exist.
- `check_availability`: check the availability of the environment manager (e.g., `conda`, `venv`).
- `exists`: check if an environment exists.
- `install_py_deps`: install Python dependencies. Usually support three ways: a "requirements.txt" file path, a dependency list, or a directory path to a library code base.
- `run_cmd`: run a command in this environment.

Now we provide two concrete implementations of `Env`:
- `CondaEnv`: use `conda` or `mamba` to manage environments.
- `VirtualEnv`: use `venv`, `virtualenv`, or `uv venv` to manage environments.

When initializing the environment manager, we can specify the environment manager to use by setting the `env_manager` parameter in the configuration file and the name of the environment by setting the `env_name` parameter. An example of the basic usage is as follows:
```python
from data_juicer_sandbox.env_manager import ENV_ROUTER

env_manager = 'conda'
env_name = 'new_conda_env'

# create an environment
env = ENV_ROUTER[env_manager](
    env_name=env_name,
    env_manager=env_manager)
# check the availability
if not env.check_availability():
    # this env manager is not available
    exit()
# create a new env. If it's already existing, use the existing one
env.create()

# install extra dependencies
# use a "requirements.txt" file
env.install_py_deps("/path/to/requirements.txt")
# use a dependency list
env.install_py_deps(["torch", "torchvision"])
# use a directory path to a library code base, e.g., InternVL
env.install_py_deps("/path/to/a/third-party/library")

# run a command in this environment
cmd = "python train.py"
env.run_cmd(cmd)
```

A complete example of using the environment manager in a hook is available in the `InternVLCOCOCaptionEvaluator` class [here](https://github.com/datajuicer/data-juicer-sandbox/blob/main/data_juicer_sandbox/specific_hooks/intervl_coco_captioning/model_hooks.py).
