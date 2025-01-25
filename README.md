# Llama-Medicalqa

I finetuned unsloth/Llama-3.2-3B-Instruct model from unsloth on MedicalQa dataset to perform effectively on medical based question.

dataset = https://huggingface.co/datasets/openchat/cogstack-opengpt-sharegpt.

# Load Model-Tokenizer & Finetune
i use unsloth/Llama-3.2-3B-Instruct from unsloth,set max seq to 2048,and load in 4bit to true.

finetuned using lora,i use rank 16,target modules q,k,v,o,gate,up,down,and set the bias and loftq to none.

# Format Dataset
since the dataset is in sharegpt,we need to format the dataset to llama-3.1 to make sure we train the model succesfully.

# Training
i use stftrainer,and only train on response not input.

# Result
fully trained model pushed in https://huggingface.co/bloombergs/Llama-3.2-3B-Instruct-MedicalQA.
