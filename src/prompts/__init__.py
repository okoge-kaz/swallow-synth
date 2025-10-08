from importlib import import_module


def get_prompt(stage: str, language: str = "python") -> str:
    """
    Dynamically load a prompt constant based on stage and language.

    Example:
        get_prompt("stage4", "python") -> returns PYTHON_STAGE4_PROMPT
        get_prompt("stage2", "finemath") -> returns FINEMATH_STAGE2_PROMPT

    Args:
        stage: e.g. "stage2", "stage4", "stage5", "stage8"
        language: e.g. "python", "finemath"

    Returns:
        The prompt string corresponding to the given stage/language.

    Raises:
        ValueError: if the stage or language is not supported.
        ImportError: if the corresponding module cannot be found.
        AttributeError: if the expected prompt constant does not exist.
    """
    match language:
        case "python":
            module_path = f"src.prompts.python.{stage}"
            const_name = f"PYTHON_{stage.upper()}_PROMPT"
        case "finemath":
            module_path = f"src.prompts.finemath.{stage}"
            const_name = f"FINEMATH_{stage.upper()}_PROMPT"
        case _:
            raise ValueError(f"Unsupported language: {language}")

    try:
        mod = import_module(module_path)
    except ModuleNotFoundError as e:
        raise ValueError(f"Prompt module not found for stage '{stage}' and language '{language}'") from e

    try:
        return getattr(mod, const_name)
    except AttributeError as e:
        raise ValueError(f"Prompt constant '{const_name}' not found in {module_path}") from e
