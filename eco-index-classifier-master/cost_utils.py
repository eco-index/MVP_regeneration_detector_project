MODEL_INFO = {
    "gpt-4.1": {
        "input_cost_per_million": 2.00,
        "output_cost_per_million": 8.00,
        "cache_input_cost_per_million": 0.20,
    },
    "gpt-4.1-mini": {
        "input_cost_per_million": 0.40,
        "output_cost_per_million": 1.60,
        "cache_input_cost_per_million": 0.04,
    },
    "gpt-4.1-nano": {
        "input_cost_per_million": 0.100,
        "output_cost_per_million": 0.400,
        "cache_input_cost_per_million": 0.010,
    },
}


def calculate_cost(prompt_tokens, completion_tokens, model_name, cached_prompt_tokens=0):
    """Calculate total cost for a request.

    Args:
        prompt_tokens: Number of prompt tokens used.
        completion_tokens: Number of completion tokens used.
        model_name: Key from MODEL_INFO.
        cached_prompt_tokens: Tokens from the prompt that are served from cache.
    Returns:
        Total cost in dollars.
    """
    pricing = MODEL_INFO[model_name]
    regular_prompt_tokens = prompt_tokens - cached_prompt_tokens
    input_cost = (regular_prompt_tokens / 1_000_000) * pricing["input_cost_per_million"]
    cache_cost = (cached_prompt_tokens / 1_000_000) * pricing["cache_input_cost_per_million"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output_cost_per_million"]
    return input_cost + cache_cost + output_cost
