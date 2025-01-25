# Llama-Medicalqa

I finetuned unsloth/Llama-3.2-3B-Instruct model from Unsloth on MedicalQa dataset to perform effectively on medical based question.

Dataset = https://huggingface.co/datasets/openchat/cogstack-opengpt-sharegpt.

# Load Model-Tokenizer & Finetune
I use unsloth/Llama-3.2-3B-Instruct from unsloth,set max seq to 2048,and load in 4bit to true.

Finetuned using lora,i use rank 16,target modules q,k,v,o,gate,up,down,and set the bias and loftq to none.

# Format Dataset
Since the dataset is in sharegpt,we need to format the dataset to llama-3.1 to make sure we train the model succesfully.

# Training
I use sfttrainer,trained for 200 steps,and only train on response not input.

# Result
Fully trained Model pushed and available in https://huggingface.co/bloombergs/Llama-3.2-3B-Instruct-MedicalQA.
