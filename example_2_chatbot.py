##############################################################
########### EXAMPLE 2 - building a simple chatbot ############
##############################################################

from transformers import AutoTokenizer, LlamaForCausalLM
from huggingface_hub import login
import utils

# Log in to Huggingface using your access token
# If you're using the console to log in you can remove this line
login(token="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")

# Choose the model you want to use
# Be careful to only use models ending in 'hf' if using Huggingface
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# Setting the model's tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Set system prompt
# This is the official 'system guide' used by LLaMA2 during training,
# you can experiment with changing the system's role/task description 
# but know it will likely impact results
system_role = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
utils.start_chat(model, tokenizer, system_role)