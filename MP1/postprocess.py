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
    functions = split_multiple_functions(response)
    for func in functions:
        if func.startswith(prompt):
            return func