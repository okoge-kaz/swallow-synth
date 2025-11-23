from importlib import import_module


def get_prompt(stage: str, language: str = "python") -> str:
    """
    Dynamically load a prompt constant based on stage and language.

    Example:
        get_prompt("stage3", "python") -> returns PYTHON_STAGE3_PROMPT
        get_prompt("stage2", "finemath") -> returns FINEMATH_STAGE2_PROMPT

    Args:
        stage: e.g. "stage3", "stage4", "stage6"
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
        case "c":
            module_path = f"src.prompts.c.{stage}"
            const_name = f"C_{stage.upper()}_PROMPT"
        case "cpp":
            module_path = f"src.prompts.cpp.{stage}"
            const_name = f"CPP_{stage.upper()}_PROMPT"
        case "cuda":
            module_path = f"src.prompts.cuda.{stage}"
            const_name = f"CUDA_{stage.upper()}_PROMPT"
        case "go":
            module_path = f"src.prompts.go.{stage}"
            const_name = f"GO_{stage.upper()}_PROMPT"
        case "rust":
            module_path = f"src.prompts.rust.{stage}"
            const_name = f"RUST_{stage.upper()}_PROMPT"
        case "javascript":
            module_path = f"src.prompts.javascript.{stage}"
            const_name = f"JAVASCRIPT_{stage.upper()}_PROMPT"
        case "typescript":
            module_path = f"src.prompts.typescript.{stage}"
            const_name = f"TYPESCRIPT_{stage.upper()}_PROMPT"
        case "nemotron_post_training_v1":
            module_path = f"src.prompts.nemotron_post_training_v1.{stage}"
            const_name = f"NEMOTRON_POST_TRAINING_V1_{stage.upper()}_PROMPT"
        case "nemotron_post_training_v1_ja":
            module_path = f"src.prompts.nemotron_post_training_v1_ja.{stage}"
            const_name = f"NEMOTRON_POST_TRAINING_V1_JA_{stage.upper()}_PROMPT"
        case "translate":
            module_path = f"src.prompts.translate.{stage}"
            const_name = f"TRANSLATE_{stage.upper()}_PROMPT"
        case "translate_code":
            module_path = f"src.prompts.translate_code.{stage}"
            const_name = f"TRANSLATE_CODE_{stage.upper()}_PROMPT"
        case "translate_science":
            module_path = f"src.prompts.translate_science.{stage}"
            const_name = f"TRANSLATE_SCIENCE_{stage.upper()}_PROMPT"
        case "medical":
            module_path = f"src.prompts.medical.{stage}"
            const_name = f"MEDICAL_{stage.upper()}_PROMPT"
        case "pdf_ja":
            module_path = f"src.prompts.pdf_ja.{stage}"
            const_name = f"PDF_JA_{stage.upper()}_PROMPT"
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
