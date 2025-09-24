def split_multiple_functions(response):
    """
    Splits the response into multiple functions if there are multiple functions defined.
    """
    functions = response.split("\n\n")
    return [func.strip() for func in functions if func.strip().startswith("def ")]

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