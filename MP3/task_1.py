import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import subprocess
import tempfile

#####################################################
# Please finish all TODOs in this file for MP3/task_1;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content) 
    
def evaluate_java(java_code, java_test, task_id):
    if not (java_test and "public class Main" in java_test):
        return False
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = (
                "import java.util.*;\n"
                "import java.lang.*;\n"
                "import java.math.*;\n"
                "import java.security.*;\n"
                "\n"
                f"{java_code}\n\n{java_test}\n"
            )
            open(os.path.join(tmpdir, "Main.java"), "w").write(src)
            if subprocess.run(["javac", "Main.java"], cwd=tmpdir).returncode != 0:
                return False
            return subprocess.run(["java", "Main"], cwd=tmpdir).returncode == 0
    except Exception as e:
        print(f"Error evaluating task {task_id}: {e}")
        return False



def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True, target_dataset=None):
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
    

    java_tests_map = {}
    java_decl_map = {}
    if target_dataset is not None:
        for j in target_dataset:
            tid = j.get("task_id", "")
            key = tid.split("/")[-1] if "/" in tid else tid
            java_tests_map[key] = j.get("test", "")
            java_decl_map[key] = j.get("declaration", "")

    results = []
    for entry in dataset:
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        python_code = (entry.get("prompt", "") or "") + (entry.get("canonical_solution", "") or "")

        task_id_raw = entry.get("task_id", "")
        task_key = task_id_raw.split("/")[-1] if "/" in task_id_raw else task_id_raw
        java_decl = java_decl_map.get(task_key, "")

        if vanilla:
            prompt =f"""You are an AI programming assistant utilizing the DeepSeek Coder model, developed by DeepSeek Company, 
and you only answer questions related to computer science. For politically sensitive questions, 
security and privacy issues, and other non-computer science questions, you will refuse to answer.

### Instruction:
Can you translate the following Python code into Java?
The new Java code must be enclosed between [Java Start] and [Java End]
{python_code}
### Response:"""
            
        else:
            prompt = f"""You are an AI programming assistant utilizing the DeepSeek Coder model, developed by DeepSeek Company, 
and you only answer questions related to computer science. For politically sensitive questions, 
security and privacy issues, and other non-computer science questions, you will refuse to answer.

### Instruction:
Can you translate the following Python code into Java?
The new Java code must be enclosed between [Java Start] and [Java End]

### Requirements
- Output ONLY Java code, no explanations.
- Do NOT use Markdown code fences (no ```java, no ```).
- Do NOT define any class named Main.
- Define a class Solution and put all translated methods inside it.
- The final output MUST be exactly in this format:

[Java Start]
<your Java code for class Solution only>
[Java End]

### Python code:
{python_code}

### Response:
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):]

        response = decoded.strip()
        java_code = response
        start_tag = "[Java Start]"
        end_tag = "[Java End]"
        if start_tag in response and end_tag in response:
            java_code = response.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()


        if "public class Solution" in java_code:
            java_code = java_code.replace("public class Solution", "class Solution")

        if "class Solution" not in java_code:
            java_code = "class Solution {\n" + java_code + "\n}\n"


        java_test = entry["test"] if "test" in entry else ""

        task_id_raw = entry.get("task_id", "")
        task_key = task_id_raw.split("/")[-1] if "/" in task_id_raw else task_id_raw

        if task_key in java_tests_map:
            java_test = java_tests_map[task_key]
        else:
            java_test = entry["test"] if "test" in entry else ""
        
        verdict = evaluate_java(java_code, java_test, entry.get("task_id", ""))


        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_expected:\n{verdict}")
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
    This Python script is to run prompt LLMs for code translation.
    Usage:
    `python3 task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

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
    target_dataset = None
    if "python" in input_dataset and "java" not in input_dataset:
        java_dataset_path = input_dataset.replace("python", "java")
        if os.path.exists(java_dataset_path):
            target_dataset = read_jsonl(java_dataset_path)
        else:
            print(f"Warning: Java dataset file {java_dataset_path} not found. Falling back to tests in {input_dataset}.")

    results = prompt_model(dataset, model, vanilla, target_dataset)
    write_jsonl(results, output_file)
