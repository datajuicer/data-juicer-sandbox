# Q&A

1. 训练 InternVL 时发生 `RuntimeError`：

```text
RuntimeError: 
        CUDA Setup failed despite GPU being available. Please run the following command to get more information:

        python -m bitsandbytes

        Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
        to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
        and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues
```
- 原因：这可能是由于 CUDA、PyTorch 和 bitsandbytes 不兼容造成的。运行 `python -m bitsandbytes` 获取更多详细信息。
- 解决方案：
  - 移除 InternVL 主目录下 `requirements/internvl_chat.txt` 中对 bitsandbytes 的版本限制，以避免在启动环境时再次安装错误版本。然后使用 `pip uninstall bitsandbytes && pip install bitsandbytes` 重新安装。
  - 如果上述解决方案无效，请重新安装与您的 GPU 的 CUDA 版本兼容的 PyTorch，并重复上述步骤，直到 `python -m bitsandbytes` 命令输出 SUCCESS。
  - 然后，还需要重新安装 `flash-attn`。

2. 在训练 InternVL 时发生 `AssertionError`：

```text
AssertionError: It is illegal to call Engine.step() inside no_sync context manager
```

- 解决方案：将 `deepspeed` 版本降级至 `0.15.4`，并在 InternVL 主目录中的 `requirements/internvl_chat.txt` 和 `pyproject.toml` 中移除 `deepspeed` 的版本限制。

3. 评测 InternVL 时报错 `java not found`：
- 解决方案：安装 java。

