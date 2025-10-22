import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import re
import json

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)


def generate_coverage(task_id, program_str, response):
    if not os.path.exists("Codes"):
        os.makedirs("Codes")
    init_path = os.path.join("Codes", "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            pass

    # Add current directory to sys.path to make 'Codes' importable
    sys.path.insert(0, os.path.abspath("."))

    suffix = "vanilla" if vanilla else "crafted"
    task_id = task_id.replace("/", "_")
    response = f"from Codes.{task_id} import *\n" + response

    save_file(program_str, f"Codes/{task_id}.py")
    save_file(response, f"Codes/{task_id}_test.py")
    report_path = f"Coverage/{task_id}_test_{suffix}.json"
    exit_code = os.system(f"pytest Codes/{task_id}_test.py --cov Codes.{task_id} --cov-report json:{report_path}")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)
        return round(report["totals"]["percent_covered"], 2)
    else:
        with open(report_path, "w") as f:
            json.dump({"totals": {"percent_covered": 0}}, f)
        return 0

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
    
    results = []
    for entry in dataset:
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        TASK_PROMPT = entry['prompt']
        program_str = entry['canonical_solution']

        if vanilla:
            prompt = f"""You are an AI programming assistant. You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
Generate a pytest test suite for the following code.
Only write unit tests in the output and nothing else.
{TASK_PROMPT + "\n" + program_str}

### Response:
"""
        else:
            prompt = f"""You are an AI programming assistant that is skilled in testcase generation. You are designed to generate testcases for the given code with high coverage (try your best to cover all the code).
### Workflow:
1. Read the task prompt and the code carefully
2. Generate a pytest test suite for the given code, including multiple test functions to cover all the code.

### Constraints:
- You must generate multiple test functions to cover all the code.
- **CRITICAL**: Only write unit tests in the output and nothing else, no other text or comments.

### Instructions:
- You are supposed to generate **NOT ONLY ONE** test function, but multiple test functions to cover all the code.
- DO NOT write many assertions in one test function, but write MULTIPLE test functions with MINIMAL assertions.
- Make sure you review the code line by line after you write it, don't use functions that are NOT defined or external.

### Important Rules:
- **Base all expected results on the actual implemented logic in the code**; when unsure, **walk through the code line-by-line** and derive outputs. **Do not invent semantics** beyond code/task description.
- Cover: typical, boundary/degenerate (empty/zero/min/max/singleton/large/negative), type variations within spec, rare branches/early returns, and **error/exception** paths where applicable.
- Follow the input constraints from the task exactly.  
- Do NOT create inputs outside the spec (e.g., if the prompt says "letters and spaces", avoid digits or punctuation).

### Task Prompt and Code:
{TASK_PROMPT + "\n" + program_str}

### Response:"""
        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=3000,
                do_sample=False,
                # temperature=0.0, No need when do_sample=False
            )
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        # parse ```python ... ``` to get the response if it exists, else use the raw response
        matches = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
        if matches:
            response = matches.group(1).strip()
        lines = response.split("\n")
        response = "\n".join([line for line in lines if not line.startswith("from")])

        # TODO: process the response, generate coverage and save it to results
        coverage = generate_coverage(entry['task_id'], TASK_PROMPT + "\n" + program_str, response)

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\ncoverage:\n{coverage}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "coverage": coverage
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
    This Python script is to run prompt LLMs for code synthesis.
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




