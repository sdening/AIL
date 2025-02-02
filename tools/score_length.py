def score_prompt_length(modified_prompt_text: str, initial_prompt: str) -> int:
    modified_length = len(modified_prompt_text)
    initial_length = len(initial_prompt)
    par = 1    
    # Calculate the ratio of modified length to initial length
    length_ratio = modified_length / par*initial_length
    
    # If the ratio is less than or equal to 1, score is 100
    if length_ratio <= 1:
        score = 100
    else:
        # If the ratio is greater than 1, decrease the score
        # Upperlimit for modified prompt text?
        score = max(0, 100 - int((length_ratio - 1) * 100))
    
    return score

def score_prompt_length_absolut(modified_prompt_text: str, initial_prompt: str) -> int:
    modified_length = len(modified_prompt_text)
    initial_length = len(initial_prompt)
    par = 1
    # If the ratio is less than or equal to 1, score is 100
    if modified_length <= par*initial_length:
        score = 100
    else:
        # If the ratio is greater than 1, decrease the score
        # Upperlimit for modified prompt text?
        score = max(0, 100 - modified_length + par*initial_length)
    
    return score

