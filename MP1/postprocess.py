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
    match = re.search(r'def\s+\w+\s*\(.*?\):', prompt)
    if not match:
        return None

    clean_prompt = match.group(0)

    functions = split_multiple_functions(response)
    for func in functions:
        # if the func string contains the clean_prompt, return it
        if clean_prompt in func:
            return func
    return None