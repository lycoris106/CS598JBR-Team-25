import jsonlines
import sys
import torch
import re
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

import re
import random

def extract_test_case(test_code):
    """
    Randomly selects one assertion like: assert candidate(...) == something
    from a HumanEval test string. Returns (input_str, expected_output_str, chosen_index).
    """
    pattern = r"assert\s+candidate\s*\((.*?)\)\s*==\s*(.+?)(?:\s*#.*)?$"
    matches = list(re.finditer(pattern, test_code, flags=re.MULTILINE))

    if not matches:
        return None, None, None

    # Randomly pick one assertion index
    chosen_idx = random.randrange(len(matches))
    m = matches[chosen_idx]

    input_str = m.group(1).strip()
    rhs = m.group(2).strip()

    # Identify output type based on first non-space character
    if rhs.startswith('['):
        output_match = re.search(r"\[.*?\]", rhs)
    elif rhs.startswith('('):
        output_match = re.search(r"\(.*?\)", rhs)
    else:
        output_match = re.search(r"([^,\s\n]+)", rhs)

    expected_output_str = output_match.group(0).strip() if output_match else None
    return input_str, expected_output_str, chosen_idx


def extract_additional_test_cases(test_code, exclude_idx=None, max_pairs=5):
    """
    Returns up to `max_pairs` (input_str, expected_output_str) pairs from assertions,
    *excluding* the one specified by exclude_idx (the randomly chosen test).
    """
    pattern = r"assert\s+candidate\s*\((.*?)\)\s*==\s*(.+?)(?:\s*#.*)?$"
    matches = list(re.finditer(pattern, test_code, flags=re.MULTILINE))

    if not matches:
        return []

    pairs = []
    for i, m in enumerate(matches):
        if i == exclude_idx:
            continue  # skip the randomly chosen one

        input_str = m.group(1).strip()
        rhs = m.group(2).strip()

        if rhs.startswith('['):
            out_m = re.search(r"\[.*?\]", rhs)
        elif rhs.startswith('('):
            out_m = re.search(r"\(.*?\)", rhs)
        else:
            out_m = re.search(r"([^,\s\n]+)", rhs)

        expected_output_str = out_m.group(0).strip() if out_m else None
        if expected_output_str:
            pairs.append((input_str, expected_output_str))
        if len(pairs) >= max_pairs:
            break

    return pairs



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
        
            input_str, expected_output_str, _ = extract_test_case(entry['test'])

            
            task_prompt = entry['prompt']
            program_str = entry['canonical_solution']
            prompt = f"""
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. 
For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

### Instruction:

If the input is {input_str}, what will the following code return?

The return value prediction must be enclosed between [Output] and [/Output] tags. For example: [Output]prediction[/Output].

{program_str}

### Response:
"""
        else:
            # advanced prompt crafting
            input_str, expected_output_str, chosen_idx = extract_test_case(entry['test'])
            few_shot_pairs = extract_additional_test_cases(entry['test'], exclude_idx=chosen_idx, max_pairs=5)


            task_prompt = entry['prompt']
            task_desc = textwrap.dedent(task_prompt).strip() 
            program_str = entry['canonical_solution']
            
            examples_block_lines = []
            for i_s, o_s in few_shot_pairs:
                examples_block_lines.append(
                    f"[Input]{i_s}[/Input]\n"
                    f"Reasoning:\n"
                    f"(explain briefly how the code transforms the input to produce the output)\n"
                    f"[Output]{o_s}[/Output]"
                )
            examples_block = "\n\n".join(examples_block_lines)

            # If no extra assertions found, we still run CoT but without ICL exemplars
            examples_section = (
                f"### In-Context Examples\n{examples_block}\n\n"
                if examples_block else
                "### In-Context Examples\n(none available from tests)\n\n"
            )

            prompt = f"""
You are an AI programming assistant that predicts the return value of a Python function
by mentally executing the code. Follow the format exactly and preserve Python literal
syntax (lists [], tuples (), dicts {{}}, strings with quotes, etc.). Do not invent new
variables. Do not modify the code. Base your reasoning on the task description and the code.

Rules:
1) Under 'Reasoning:', explain step by step how the code runs on the given input.
2) End with a single pair of [Output]...[/Output] containing ONLY the final return value.
3) If the true output is a list/tuple/dict/string, keep brackets/parentheses/quotes exactly.
4) Output nothing besides the 'Reasoning:' section and the final [Output] block.

### Task (verbatim from dataset)
{task_desc}

### Code under analysis
{program_str}

{examples_section}### Now solve the following in the same format
[Input]{input_str}[/Input]
Reasoning:
"""

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
        pred_match = re.search(r"\[Output\](.*?)\[/Output\]", response, re.DOTALL)
        pred_output = pred_match.group(1).strip() if pred_match else None
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