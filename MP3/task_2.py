import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP3/task_2;
#####################################################

import re
import random

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def parse_response(response):
    patterns = [
        r'<verdict>\s*(Buggy|Correct)\s*</verdict>',
        r'<verdict>\s*(buggy|correct)\s*</verdict>',
        r'<verdict>\s*(Buggy|Correct)\s*<verdict>',
        r'<verdict>\s*(buggy|correct)\s*<verdict>',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            verdict = match.group(1).lower()
            return verdict == "buggy"
    
    # Fallback: check if "buggy" appears more prominently in conclusion
    response_lower = response.lower()
    
    buggy_count = response_lower.count("buggy")
    correct_count = response_lower.count("correct")
    
    if buggy_count > correct_count:
        return True
    
    return False

def create_vanilla_prompt(entry):
    prompt = f"""You are an AI programming assistant. You are an AI programming assistant utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

### Instruction:

The following is the problem statement and function signature:
{entry['prompt']}

{entry['buggy_solution']}

Is the above code buggy or correct? Please explain your step by step reasoning. The prediction should be enclosed within <verdict> and </verdict> tags. For example: <verdict>Buggy</verdict>

### Response:"""
    return prompt

def create_crafted_prompt(entry):
    prompt = f"""You are an expert Python developer specializing in bug finding.
Your job is to decide whether the given implementation is BUGGY or CORRECT.

### Specification
The following is the problem statement and function signature:
{entry['prompt']}

### Candidate Implementation

{entry['buggy_solution']}

### Instructions

You MUST be skeptical: even if the code looks reasonable, carefully check for subtle mistakes:
1. Is each usage of literal number or variable correct? 
2. Is there any operator that is misused?
3. Is each condition following the logic in specification? Is it forgetting to check certain conditions?
If any of the above checks fails, you should consider this as buggy.

**Critical**: The prediction should be enclosed within <verdict> and </verdict> tags. For example: <verdict>Buggy</verdict>
**Critical**: Do not try to fix the implementation. If the implementation is buggy, just output <verdict>Buggy</verdict>.

### Response:"""
    return prompt

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
    print(f"Working with {model_name} prompt type {vanilla}...")
    
    # TODO: download the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # TODO: load the model with quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,     
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    random.seed(4321)
    
    results = []
    for entry in dataset:
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        prompt = ""

        if vanilla:
            prompt = create_vanilla_prompt(entry)
        else:
            prompt = create_crafted_prompt(entry)
        
        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                # temperature=0.0, No need when do_sample=False
            )
        # Remove the prompt part 
        gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # TODO: process the response and save it to results
        predicted_buggy = parse_response(response)
        expected_buggy = True
        is_correct = (predicted_buggy == expected_buggy)

        print(f"\nTask_ID {entry['task_id']}:")
        print(f"Predicted: {'Buggy' if predicted_buggy else 'Correct'}")
        print(f"Expected: {'Buggy' if expected_buggy else 'Correct'}")
        print(f"Is Correct: {is_correct}")
        print("-" * 80)

        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "is_correct": is_correct
        })
        
    return results

def read_jsonl(file_path):
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader: 
            dataset.append(line)
    return dataset

def write_jsonl(results, file_path):
    with jsonlines.open(file_path, "w") as f:
        for item in results:
            f.write_all([item])

if __name__ == "__main__":
    """
    This Python script is to run prompt LLMs for bug detection.
    Usage:
    `python3 task_2.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

    Inputs:
    - <input_dataset>: A `.jsonl` file, which should be your team's dataset containing 20 HumanEval problems.
    - <model>: Specify the model to use. Options are "deepseek-ai/deepseek-coder-6.7b-base" or "deepseek-ai/deepseek-coder-6.7b-instruct".
    - <output_file>: A `.jsonl` file where the results will be saved.
    - <if_vanilla>: Set to 'True' or 'False' to enable vanilla prompt
    
    Outputs:
    - You can check <output_file> for detailed information.
    """
    args = sys.argv[1:]
    input_dataset = args[0]
    model = args[1]
    output_file = args[2]
    if_vanilla = args[3] # True or False
    
    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")
    
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")
    
    vanilla = True if if_vanilla == "True" else False
    
    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, vanilla)
    write_jsonl(results, output_file)
