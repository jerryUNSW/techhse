import os

def load_system_prompt(prompt_name="system_prompt.txt"):
    """
    Load system prompt from the prompts directory.
    
    Args:
        prompt_name (str): Name of the prompt file (default: system_prompt.txt)
    
    Returns:
        str: The system prompt content
    """
    prompt_path = os.path.join("prompts", prompt_name)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback to resolve relative to repository root (this file's directory)
        module_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(module_dir, "prompts", prompt_name)
        try:
            with open(alt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
    except Exception as e:
        raise Exception(f"Error loading system prompt: {e}")

def load_user_prompt_template(template_name="user_prompt_template.txt"):
    """
    Load user prompt template from the prompts directory.
    
    Args:
        template_name (str): Name of the template file (default: user_prompt_template.txt)
    
    Returns:
        str: The user prompt template content
    """
    template_path = os.path.join("prompts", template_name)
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback to resolve relative to repository root (this file's directory)
        module_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(module_dir, "prompts", template_name)
        try:
            with open(alt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"User prompt template file not found: {template_path}")
    except Exception as e:
        raise Exception(f"Error loading user prompt template: {e}")

def format_user_prompt(template, **kwargs):
    """
    Format a user prompt template with the provided arguments.
    
    Args:
        template (str): The prompt template
        **kwargs: Arguments to format into the template
    
    Returns:
        str: The formatted user prompt
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise KeyError(f"Missing required argument in prompt template: {e}")
    except Exception as e:
        raise Exception(f"Error formatting user prompt: {e}")
