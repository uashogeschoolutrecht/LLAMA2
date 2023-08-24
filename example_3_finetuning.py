##############################################################
########### EXAMPLE 3 - Finetuning a LLaMA2 model ############
##############################################################

"""
########## RTFM! ##########

(also known as read the README.md)
"""

import utils as utils
from huggingface_hub import login
from transformers import AutoModelForCausalLM

# Log in to Huggingface using your access token
login(token="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")

# Set the directory where you want model checkpoints to be saved to
output_dir = "./results_example_3"

# Retrieve the dataset and configurations
dataset = utils.get_dataset("timdettmers/openassistant-guanaco")
bnb_config, peft_config, training_config = utils.get_configs(output_dir)

# Get the model and corresponding tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
model, tokenizer = utils.get_model_and_tokenizer(model_name, bnb_config)

# Train and save the model
utils.train_model(model, dataset, tokenizer, peft_config, training_config)

# Load the new model
merged_model_path = "results_example_3/final_checkpoint/final_merged_checkpoint"
new_model = utils.load_new_model(merged_model_path)

# Get the configuration used for generating text
gen_config = utils.get_generation_config()

# Prompt the model
prompt = "What is rain?"
response = utils.prompt_model(prompt, new_model, tokenizer, gen_config)
print(response)

# Chat with the model
# Warning, for chat you should use a (finetuned) 'chat' version of LLaMA 
# e.g., meta-llama/Llama-2-7b-chat-hf NOT meta-llama/Llama-2-7b-hf
system_role = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
utils.start_chat(new_model, tokenizer, system_role, max_tokens = 2048)