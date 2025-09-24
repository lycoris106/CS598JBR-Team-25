import re

def split_multiple_functions(code):
    triple_string_pattern = r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')'
    placeholders = {}
    
    def replacer(match):
        key = f"__PLACEHOLDER_{len(placeholders)}__"
        placeholders[key] = match.group(0)
        return key

    protected_code = re.sub(triple_string_pattern, replacer, code)
    raw_blocks = protected_code.split("\n\n\n")

    restored_blocks = [
        re.sub("|".join(map(re.escape, placeholders.keys())),
               lambda m: placeholders[m.group(0)],
               block)
        for block in raw_blocks
    ]

    return [b.strip() for b in restored_blocks if b.strip()]

def filter_response(prompt, response):
    """
    Filters out any extraneous text before the first function definition.
    """
    # find the first function in prompt using regex
    import re
    match = re.search(r'def\s+\w+\s*\(.*?\)\s*(->\s*\w+)?\s*:'
, prompt)
    if not match:
        return None

    clean_prompt = match.group(0)
    functions = split_multiple_functions(response)
    for func in functions:
        # if the func string contains the clean_prompt, return it
        if clean_prompt in func:
            return func
    return None

def get_completion(prompt, response):
    response = filter_response(prompt, response)
    if response:
        # remove the definition line from the response
        lines = response.split("\n")
        if len(lines) > 1:
            lines = lines[1:]
        
        # remove the comments/docstring if any (including the content)
        while lines and (lines[0].strip().startswith('"""') or lines[0].strip().startswith("'''")):
            lines.pop(0)
            while lines and not (lines[0].strip().endswith('"""') or lines[0].strip().endswith("'''")):
                lines.pop(0)
            if lines:
                lines.pop(0)  # remove the ending docstring line
        
        return "\n".join(lines)
    return None