# Q&A

1. `RuntimeError` when training InternVL:

```text
RuntimeError: 
        CUDA Setup failed despite GPU being available. Please run the following command to get more information:

        python -m bitsandbytes

        Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
        to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
        and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues
```
- Reason: it might be the reason of incompatibility of CUDA, PyTorch, and bitsandbytes. Run `python -m bitsandbytes` for more details.
- Solution:
  - Remove the version limitation of bitsandbytes in `requirements/internvl_chat.txt` in the home of InternVL to avoid installing the wrong version again when starting the env. Then reinstall it with `pip uninstall bitsandbytes && pip install bitsandbytes`.
  - If the above solution does not work, reinstall the PyTorch that is compatible with the CUDA version of your GPU, and repeat the above step, until the command `python -m bitsandbytes` outputs SUCCESS.
  - Then, the `flash-attn` needs to be reinstalled as well.

2. `AssertionError` when training InternVL:

```text
AssertionError: It is illegal to call Engine.step() inside no_sync context manager
```
- Solution: downgrade the version of `deepspeed` to `0.15.4`, and remove the version limitation of `deepspeed` in both `requirements/internvl_chat.txt` and `pyproject.toml` in the home of InternVL.

3. `java not found` when evaluating InternVL:
- Solution: install java.
