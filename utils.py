"""
This file contains utility functions used throughout the examples.

You can adjust any parameters and/or model configurations here if you wish to do so,
but please be careful and read up on the changes you make. Any small deviation can
have quite severe effects on the performance/behaviour of the model (or just make it
crash altogether). Other than that though, experimentation is of course why we made
this easily accessible in the first place, just be sure to do your due diligence.
"""

import os
import torch
from typing import Union
import datasets
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, LlamaForCausalLM, LlamaTokenizer, LlamaTokenizerFast, GenerationConfig, Pipeline
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel
from trl import SFTTrainer
from huggingface_hub import login

def get_dataset(dataset_name : str) -> datasets.Dataset:
    """
    Retrieves the dataset you want to use for finetuning from HuggingFace Hub. If you 
    want to use your own custom dataset, it's easiest to place it on HuggingFace Hub
    such that others are also able to use your dataset (part of the open-source plan).

    dataset_name [str] = name of the dataset hosted on HuggingFace Hub you want to load in

    returns a datasets.Dataset object
    """

    print(f"########## Loading HuggingFace Hub dataset: {dataset_name} ##########")

    return( datasets.load_dataset(dataset_name, split="train") )


def get_local_dataset(file_type : str, file_path) -> datasets.Dataset:
    """
    Retrieves the dataset you want to use for finetuning from a local directory.
    Check https://huggingface.co/docs/datasets/v1.2.1/loading_datasets.html for more details.

    file_type [str]: what file type is the dataset you're loading in (e.g., 'csv', 'json', 'parquet')
    file_path [str/list]: path to file, including filename. if multiple files you can give a list.

    returns a datasets.Dataset object

    examples:
    get_local_dataset("csv", "training_data.csv")
    get_local_dataset("csv", ["training_data_1.csv", "training_data_2.csv"])
    get_local_dataset("json", "./data/datasets/training_data.json")

    """

    print(f"########## Loading local dataset file in {file_type} format from {file_path} ##########")

    return( datasets.load_dataset(file_type, data_files=file_path) )

def get_configs(output_dir : str):
    """
    Returns configurations for PEFT, bitsandbytes, and training.

    output_dir [str]: location to place the output of training (model checkpoints)
    """

    print(f"########## Retrieving configurations for bnb, peft, and training. ##########")
    print(f"########## Model checkpoints will be saved to: {output_dir} ##########")

    bnb_config = get_bnb_config()

    peft_config = get_peft_config()

    training_config = get_training_config(output_dir)

    return(bnb_config, peft_config, training_config)

def get_bnb_config() -> BitsAndBytesConfig:
    """
    Returns the configuration for bitsandbytes, which is the package
    used for quantization.
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return(bnb_config)

def get_peft_config() -> LoraConfig:
    """
    Returns the configuration for Parameter-Efficient FineTuning (PEFT).
    """

    peft_config = LoraConfig(
        lora_alpha=16, #parameter for Lora scaling
        lora_dropout=0.1, #dropout probability for Lora layers
        r=64, #dimension of the updated matrices
        bias="none",#bias to induce, generally none
        task_type="CAUSAL_LM" #the LM 'type' we are using, what task to ask it to do
    )

    return(peft_config)

def get_training_config(output_dir : str) -> TrainingArguments:
    """
    Returns the configuration for the training phase of the finetuning process.

    output_dir [str]: location to place the output of training (model checkpoints)
    """

    training_config = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=500

        # More parameters you can tune if wanted:
        # optim=optim,
        # save_steps=save_steps,
        # fp16=True/False,
        # bf16=True/False,
        # max_grad_norm=max_grad_norm,
        # warmup_ratio=warmup_ratio,
        # group_by_length=group_by_length,
        # lr_scheduler_type=lr_scheduler_type
        # num_train_epochs=num_train_epochs
        # weight_decay = weight_decay
    )

    return(training_config)   

def get_generation_config() -> GenerationConfig:
    """
    Returns the configuration used for generating output. You can also
    modify this function to get the configuration from a local file,
    thus easily creating various templates you want to use/compare to
    eachother.

    If you remove variables from the GenerationConfig, this just means
    the model will use default values, so it's relatively safe to do.

    See for full list of parameters and their explanation: 
    https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig

    returns a GenerationConfig containing all parameters for generating
    output that may differ from the default value the model uses.
    """

    # Edit this to your heart's content
    gen_config = GenerationConfig(
        max_new_tokens = 200,
        temperature = 0.7,
        top_p = 0.1,
        top_k = 40,
        repetition_penalty = 1.1
    )

    return(gen_config)


def get_model_and_tokenizer(model_name : str, bnb_config : BitsAndBytesConfig) -> [LlamaForCausalLM, LlamaTokenizer]:
    """
    Returns both model and appropriate tokenizer. Currently only supporting
    LLaMA-2 models. LLaMA-1 models may work too but haven't been tested.

    Valid model names as of writing are:
    "meta-llama/llama-2-7b-hf"
    "meta-llama/llama-2-13b-hf"
    "meta-llama/llama-2-7b-chat-hf"
    "meta-llama/llama-2-13b-chat-hf"

    If you want to use the 70B models of LLaMA2, get in contact with SURF, as
    they necessitate multiple GPU's to run and are much more fragile in their
    configuration (and much more costly to experiment with).

    model_name [str]: name of the HuggingFace model you want to use.
    """

    print(f"########## Retrieving model: {model_name} ##########")
    model = get_model(model_name, bnb_config)

    print(f"########## Retrieving tokenizer for model: {model_name} ##########")
    tokenizer = get_tokenizer(model_name)

    return(model, tokenizer)

def get_model(model_name : str, bnb_config : BitsAndBytesConfig) -> LlamaForCausalLM:
    """
    Retrieve the model from HuggingFace Hub using model_name. Integrates
    the bnb_config for quantization.

    model_name [str]: name of the HuggingFace model you want to use.
    bnb_config: the bitsandbytes configuration for quantization.
    """

    # Load the entire model on 'GPU 0' (the only one available)
    # switch to `device_map = "auto"` for multi-GPU
    device_map = {"": 0}

    # Load in the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True, #redundant? also used elsewhere in lower level
        use_auth_token=True
    )
    model.config.use_cache = False

    # More info: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1

    return(model)

def get_tokenizer(model_name : str) -> LlamaTokenizer:
    """
    Retrieve the appropriate tokenizer fitting the HuggingFace model.

    model_name [str]: name of the HuggingFace model you want to use.
    """

    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set the padding token (as LLaMA is an LLM without padding tokens, but it needs to
    # be addressed for HuggingFace training)
    # tokenizer.pad_token = tokenizer.eos_token #used a lot but models may be bad at stopping generating text
    tokenizer.pad_token = "<PAD>"
    tokenizer.padding_side = "right"

    return(tokenizer)

def train_model(model : LlamaForCausalLM, dataset : datasets.Dataset, tokenizer : Union[LlamaTokenizer, LlamaTokenizerFast], peft_config : LoraConfig, training_config : TrainingArguments, max_seq_length : int = 512) -> None:
    """
    Trains the model given all arguments. Saves the resulting model to set output_dir.

    max_seq_length [int]: how large of an input to use (constrained largely by GPU memory)
    """

    print(f"########## Starting training procedure. ##########")

    # HugginFace's (HF) Supervised Finetuning Trainer (SFT)
    # Connects directly into HF Datasets& HF Transformers
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        peft_config=peft_config,
        args=training_config,
        max_seq_length=max_seq_length
    )

    trainer.train()

    print(f"########## Saving model to {training_config.output_dir} ##########")

    save_model(trainer, training_config.output_dir)

def save_model(trainer : SFTTrainer, output_dir : str) -> None:
    """
    Saves the model checkpoints to given output directory.

    Also saves a merged-weights model, which is not strictly necessary, 
    as you get a full new LLaMA model with just a tiny amount of difference 
    but still takes up the full 10+GB of storage. But it does need to be 
    done if you want to do further finetuning on the newly made model.
    """

    # Save trained model to output_dir
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

    # Save merged-weights model
    # Free memory for merging weights
    torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)

def load_new_model(merged_model_path : str) -> LlamaForCausalLM:
    """
    Loads the final merged model created after finetuning, on default being in the folder
    'final_merged_checkpoint'. 

    returns the new model, being the old model merged with the finetuned adapters
    """

    print(f"########## Loading the merged model present at {merged_model_path} ##########")

    return(AutoModelForCausalLM.from_pretrained(merged_model_path, load_in_8bit=False))

def prompt_model(prompt : str, model : LlamaForCausalLM, tokenizer : Union[LlamaTokenizer, LlamaTokenizerFast], gen_config : GenerationConfig) -> str:
    """
    Prompt the given model with the given prompt, given a max_length for response
    length. 

    prompt [str]: the prompt to give the model
    model [LlamaForCausalLM]: model used for inference
    tokenizer [LlamaTokenizer]|[LlamaTokenizerFast]: tokenizer to use for the model inference
    max_length [int]: maximum length for the response of the model

    returns a string answer for the given prompt
    """

    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, generation_config=gen_config)
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return(response)

def prompt_model_pipeline(prompt : str, pipeline : Pipeline, tokenizer : Union[LlamaTokenizer, LlamaTokenizerFast]) -> None:
    """
    Prompt the given HuggingFace Pipeline model with the given prompt.
    """

    # input to the model + generate output
    sequences = pipeline(
        # input text (prompt) plus newline for clarity sake
        prompt + '\n',
        # parameters to use in generating output
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200
    )

    # print output
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

def start_chat(model : LlamaForCausalLM, tokenizer : Union[LlamaTokenizer, LlamaTokenizerFast], system_role : str, max_tokens = 2048) -> None:
    """
    Starts a new chat in console. Runs until context-limit is reached (set
    by max_tokens, which for LLaMA2 models can run up to ~4000 tokens) or
    till the user presses control+c. This chat keeps everything in memory
    at all times, you could for exampleadjust it to make summaries of its own 
    history after a certain threshold of tokens is used to make it last longer, 
    this is just a base to start with.

    model = the model you want to use
    tokenizer = the tokenizer that corresponds to the given model
    system_role = what task you want to give the model in the chat
    max_tokens = the maximum amount of tokens for the entire chat to fit in
    """

    # Set system prompt
    # This is the official 'system guide' used by LLaMA2 during training
    system_prompt = f"[INST] <<SYS>>\n{system_role}\n<</SYS>>"

    # Set up parameters
    history = ""

    # Start the chat
    print("Model loaded, you can now start your chat. Press control+C to stop.\n")
    while(len(tokenizer.encode(history)) < max_tokens - 200):

        # Receive input
        user_input = input()
        prompt = f"<s>{system_prompt}\n\n{history}{user_input} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate output
        generate_ids = model.generate(inputs.input_ids, max_length=max_tokens, temperature=0.1)
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        response_clean = cleanresponse(response)

        # Add user input & model output to history within defined LLaMA prompt format
        history = f"{history}{user_input} [/INST] {response_clean} </s><s>[INST] "

        print(f"\n{response_clean}\n")

def cleanresponse(response : str) -> str:
    """
    Function that cleans up the response (output) of a HuggingFace LLaMA2 model.

    Example:
    # Get input
    user_input = input()
    prompt = f"<s>{system_prompt}\n\n{history}{user_input} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate output
    generate_ids = model.generate(inputs.input_ids, max_length=2048)
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    response_clean = HF_cleanresponse(response, prompt)

    print(response_clean)
    """
    
    # grab the last question&response pair
    cleanresponse = response.split("<s>")[-1]
    # grab the last response
    cleanresponse = cleanresponse.split("[/INST]")[-1]
    # clean up
    cleanresponse = cleanresponse.replace("</s>","")

    return cleanresponse