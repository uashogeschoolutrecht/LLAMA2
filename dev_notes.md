- had to pip install:
  - transformers
  - accelerator
  - peft
  - bitsandbytes
  - datasets
  - xformers (for use in HF pipelines)
  - trl
  - huggingface_hub
  - scipy (?)
- got warning 'WARNING: The scripts f2py, f2py3 and f2py3.8 are installed in '/home/fkok2/.local/bin' which is not on PATH.' so ran command: `export PATH=/home/fkok2/.local/bin:$PATH`

- might want to do a local install of LLaMA2 converted to HuggingFace format (faster than calculating checkpoints every time.. even if it doesn't take too long to do so). Although for generalisation sake it's nice for people to get the hang already of using huggingface as it's meant (helps when they want to do research on other open access models)


just some resources for finetuning i used:
- https://blog.ovhcloud.com/fine-tuning-llama-2-models-using-a-single-gpu-qlora-and-ai-notebooks/
- https://medium.com/@ud.chandra/instruction-fine-tuning-llama-2-with-pefts-qlora-method-d6a801ebb19
- https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da

