from abc import ABC


class RewritePipeline(ABC):
    """
    Abstract base class for a rewrite pipeline.
    """

    def __init__(self, model_name: str, tensor_parallel_size: int = 1, max_model_len: int = 131072):
        """
        Initialize the rewrite pipeline.

        Args:
            model_name (str): The name of the model.
            tensor_parallel_size (int): The size of the tensor parallelism.
            max_model_len (int): The maximum length of the model.
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len

    def generate(self, prompt: list[str], **kwargs) -> list[str]:
        """
        Generate text based on the provided prompt.

        Args:
            prompt (str): The input prompt for text generation.
            **kwargs: Additional keyword arguments for generation.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def fix_errors(self, codes: list[str], lint_reports: list[str]) -> list[str]:
        """
        Fix errors in the provided codes based on lint reports.

        Args:
            codes (list[str]): The list of code snippets to fix.
            lint_reports (list[str]): The lint reports indicating errors.

        Returns:
            list[str]: The fixed code snippets.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
