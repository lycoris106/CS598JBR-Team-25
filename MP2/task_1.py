import jsonlines
import sys
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def extract_test_case(test_code):
    """
    Extracts the first input and expected output from a HumanEval test string.
    Handles list, tuple, and scalar outputs.
    """
    # Find the first assertion like: assert candidate(...) == something
    match = re.search(r"assert\s+candidate\s*\((.*?)\)\s*==\s*(.+)", test_code)
    if not match:
        return None, None

    input_str = match.group(1).strip()
    rhs = match.group(2).strip()

    # Identify output type based on first non-space character after '=='
    if rhs.startswith('['):
        # list output: everything between the first [ and the matching ]
        output_match = re.search(r"\[.*?\]", rhs)
    elif rhs.startswith('('):
        # tuple output: everything between the first ( and matching )
        output_match = re.search(r"\(.*?\)", rhs)
    else:
        # scalar: extract up to first comma, space, or newline
        output_match = re.search(r"([^,\s\n]+)", rhs)

    expected_output_str = output_match.group(0).strip() if output_match else None
    return input_str, expected_output_str

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
    
    results = []
    for entry in dataset:
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        TASK_PROMPT = entry['prompt']
        program_str = entry['canonical_solution']
        
        input_str, expected_output_str = extract_test_case(entry['test'])
        

        prompt = f"""
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. 
For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

### Instruction:

If the input is {input_str}, what will the following code return?

The return value prediction must be enclosed between [Output] and [/Output] tags. For example: [Output]prediction[/Output].

{program_str}

### Response:
"""
        
        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=500,
                do_sample=False,
                # temperature=0.0, No need when do_sample=False
            )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # TODO: process the response and save it to results
        pred_matches = re.findall(r"\[Output\](.*?)\[/Output\]", response, re.DOTALL)
        if pred_matches:
            pred_output = pred_matches[-1].strip()  # get the last occurrence
        else:
            pred_output = None
        print(f"expected_output_str: {expected_output_str}, pred_output: {pred_output}")
        
        verdict = False
        if pred_output is not None:
            try:
                # Evaluate safely: both sides as Python literals (e.g., [2,3] → list)
                import ast
                gt_value = ast.literal_eval(expected_output_str)
                pred_value = ast.literal_eval(pred_output)
                verdict = (gt_value == pred_value)
            except Exception:
                # Fallback: direct string comparison
                verdict = (pred_output == expected_output_str)

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_correct:\n{verdict}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "is_correct": verdict
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
    `python3 Task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

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