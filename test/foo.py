from dataclasses import dataclass

@dataclass
class ProcessorConfig:
    """
    Configuration for BaseProcessor.

    Attributes:
        log_level (str): The logging level.
        timeout (int): Timeout for processing tasks in seconds.
        retry (bool): Whether to retry on failure.
        max_retries (int): Maximum number of retries.
        new_param (int): An example new parameter.
    """
    log_level: str = "INFO"
    timeout: int = 30
    retry: bool = True
    max_retries: int = 3
    new_param: int = 42


class BaseProcessor:
    """
    Base processor for handling data processing tasks.

    Attributes:
        name (str): The name identifier for the processor.
        config (ProcessorConfig): Common configuration for the processor.
    """

    def __init__(self, name: str, config: ProcessorConfig) -> None:
        """
        Initialize the BaseProcessor.

        Args:
            name (str): The name of the processor.
            config (ProcessorConfig): Common configuration settings.
        """
        self.name = name
        self.config = config

    def process(self, data):
        """
        Process data. This method should be overridden by subclasses.

        Args:
            data (Any): Input data to process.

        Raises:
            NotImplementedError: If the method is not overridden.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class TextProcessor(BaseProcessor):
    """
    Processor specialized for handling text data.

    Attributes:
        language (str): The language code of the text (e.g., 'en', 'es').
    """

    def __init__(self, name: str, language: str, config: ProcessorConfig) -> None:
        """
        Initialize the TextProcessor.

        Args:
            name (str): Name of the processor.
            language (str): Language code (e.g., 'en', 'es').
            config (ProcessorConfig): Common configuration settings.
        """
        super().__init__(name, config)
        self.language = language

    def process(self, data: str) -> str:
        """
        Process text data by converting it to uppercase.

        Args:
            data (str): The input text data.

        Returns:
            str: Processed text.
        """
        # Example processing: convert text to uppercase
        return data.upper()


# Example usage:
if __name__ == '__main__':
    # Create a configuration instance
    config = ProcessorConfig(log_level="DEBUG", timeout=60, retry=False, max_retries=0, new_param=100)

    # Instantiate a text processor with its unique parameter 'language' and the common config
    text_proc = TextProcessor(name="TextProc1", language="en", log_level="DEBUG", timeout=60, retry=False, max_retries=0, new_param=100)
    print(text_proc.process("hello world"))
