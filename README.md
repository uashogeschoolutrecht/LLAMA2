The example scripts have been made by Fabian Kok, AI researcher at Hogeschool Utrecht (Lectorate of Artifical Intelligence). For questions and/or feedback, feel free to contact him at fabian.kok@hu.nl (for example if certain functions don't work anymore due to deprecated methods, tends to happen over time).

##### SET UP ########################################

Within these examples, we're using HugginFace to most easily be able
to use the newest models and various useful integrations. HuggingFace
is also an often-used platform for open-source (LLM) development so
anything you learn/use here may also help you with other (LLM) models.

In order to use HuggingFace Transformers, you first need to be logged
into your HuggingFace account with which you have gotten access to
the LLaMA2 directory. In order to get this access:

1) register for LLaMA2 access at Meta [LINK]
2) register for LLaMA2 access at HuggingFace [LINK]

*Make sure your HuggingFace and Meta mail address are the same!*

Then when running the program, you will need to supply a token, 
which you can find on your HuggingFace account under 'Access Tokens'.

example token: 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

After you've got your token, you need to paste it into the code where
you see 'login(token="hf_1234567890")' and put your own token in. This
is always located at the top of the example file.

##### HuggingFace models ########################################

Another important point when using HuggingFace's LLaMA(2) models,
is that you use the models ending in 'hf'. This is important because
natively the LLaMA models don't interact well with the HuggingFace
functions. The models ending in 'hf' have already been converted to
play by HuggingFace's rules. The advantage of this is that you can 
swap out the model for any other available HuggingFace model without
changing your codebase (too much), making it significantly easier to
test out various models and how they compare in reaction to the same
independent variable.

##### LLaMA prompt format ########################################

Within the chat versions of the LLaMA models, it is important to use
the same format when sending it prompts as what it was trained on.
No worries, this has already been tackled in the `utils.py` file's
functions, so unless you want to dive deeper into it you don't have 
to deal with this. If you do want to adapt/change it, below is the 
default format used by LLaMA(2) models so you can use that as reference.

The system prompt is the same as used by Meta itself, you can safely
adjust this to see what effect it has on the model. The double {{ }} 
marks are used to indicate variables, and would be replaced with strings
(a 'string' is what we call a piece of text in programming).

##### Setting up the example strings #####
{{ system_prompt }} = 
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

{{ user_msg_1 }} = There's a llama in my garden 😱 What should I do?

##### Format single prompt #####
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_msg_1 }} [/INST]

##### Format multiple prompts ######
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]

##### Finetuning ########################################

For finetuning, various formats exist for your input data, and they will have some effect on the outcome of training. The dataset we're using in the finetuning example is from the Guanaco finetune: https://huggingface.co/datasets/timdettmers/openassistant-guanaco. You can use various file formats, generally for HuggingFace a jsonl (json lines) format works best. If you have a different file format you can also easily use HuggingFace's datasets package to convert it. You can find in `utils.py` a function called `get_local_dataset()` which is able to load (multiple) local .csv files (and also other file formats but check which at https://huggingface.co/docs/datasets/v1.2.1/loading_datasets.html)

When doing finetuning, be sure to look carefully at the `output_dir` (the path and name of the folder you want the results to end up in, it does not need to exist yet), `model_name` (the base model you want to finetune, make sure it's an 'hf' model, and check whether it is, and whether you want, one of the chat versions), and `merged_model_path` (which is the path to the folder containing the model files) which is used to load the model.

Now, in order to run the finetuning example, you need to use the following command in the console (don't forget the `&` at the end!):

nohup python3 -u /home/{USERNAME}/example_3_finetuning.py > example_3_finetuning.log &

where {USERNAME} is the username you use in Research Cloud to login. If you don't use this command, the script will stop running once you close your connection to the server, which you are likely to do since finetuning will take >4 hours even on the small LLaMA2-7b model. The command will put all the console messages (prints, errors, etc.) in the 'example_3_finetuning.log` file (you can rename this in something else by changing it in the `nohup` command above). This updates in real-time so you can check the progress of the model in there.

As a means of reference, finetuning the 7B LLaMA model took about ~4hours in testing, the 13B LLaMA model took about ~8 hours.