##############################################################
############ EXAMPLE 1 - Running a simple prompt #############
##############################################################

from transformers import AutoTokenizer, pipeline
from huggingface_hub import login
import torch
import utils

# Log in to Huggingface using your access token
login(token="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")

# Choose the model you want to use
# Be careful to only use models ending in 'hf' if using Huggingface
model = "meta-llama/Llama-2-7b-chat-hf"

# setting the model's tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)
# set-up the model as a HuggingFace pipeline
pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto"
)

# create your prompt
prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?'
utils.prompt_model_pipeline(prompt, pipeline, tokenizer)