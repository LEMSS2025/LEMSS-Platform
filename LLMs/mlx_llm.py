from LLMs.LLM import LLM
from utils.logger import setup_logger
from constants.constants import MLX_LLM_LOG_FILE, MLX_LLM_LOG_NAME


class MLXLLM(LLM):
    """
        MLXLLM class for generating text using a specified model with configurable temperature from MLX.
        Inherits from the base LLM class.
    """

    def __init__(self, model_name: str, temperature: float, token: str, **kwargs):
        """
        Initialize the MLXLLM model with the specified model name and temperature.

        :param model_name: Name of the model to use for text generation.
        :param temperature: Temperature parameter for controlling randomness in generation.
        :param token: Token to use for the model.
        """
        super().__init__(model_name, temperature, token)

        self.__generate_flags = kwargs

        self.__logger = setup_logger(MLX_LLM_LOG_NAME, MLX_LLM_LOG_FILE)
        self.__logger.info(f"MLX LLM model initialized successfully with model: {model_name}")

        try:
            from mlx_lm import load, generate
            # Load model and tokenizer for Metal Performance Shaders (MPS) device
            self.__model, self.__tokenizer = load(model_name)
        except Exception as e:
            self.__logger.error(f"Error initializing MLX model: {e}")
            raise

    def generate_prompt(self, user: str, system: str, max_tokens: int, clean: bool = True,
                        force_max_tokens: bool = False) -> str:
        """
        Generate a text document based on user and system prompts.

        :param user: The user prompt.
        :param system: The system prompt.
        :param max_tokens: Maximum number of tokens for the generated document.
        :param clean: Whether to clean the document of extraneous text or not.
        :param force_max_tokens: Whether to manually restrict the number of tokens in the generated document.

        :return: The generated document as a string.
        """
        try:
            # Construct the message structure
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]

            # Generate text based on device type
            try:
                from mlx_lm import generate
                input_ids = self.__tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                formatted_prompt = self.__tokenizer.decode(input_ids)
                result = generate(self.__model, self.__tokenizer, formatted_prompt, max_tokens, temp=self.temperature,
                                  **self.__generate_flags)
            except Exception as e:
                # Modify the messages for the second attempt
                messages = [{"role": "user", "content": f"{system} {user}"}]

                try:
                    input_ids = self.__tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                    formatted_prompt = self.__tokenizer.decode(input_ids)
                    result = generate(self.__model, self.__tokenizer, formatted_prompt, max_tokens,
                                      temp=self.temperature, **self.__generate_flags)
                except Exception as e:
                    self.__logger.error(f"Error in generating prompt on second attempt: {e}")
                    raise

            # Clean the generated document
            if clean:
                cleaned_result = self.clean_document(result, max_tokens, self)

                # Trim the generated document to max_tokens length
                if force_max_tokens:
                    cleaned_result = self.__trim_tokens(cleaned_result, max_tokens)

                return cleaned_result, result
            else:
                return result
        except Exception as e:
            self.__logger.error(f"Error in generating prompt: {e}")
            raise

    def __trim_tokens(self, input_string: str, max_tokens: int) -> str:
        """
        Trims the input string to ensure that it contains no more than max_tokens tokens.

        :param input_string: The string to be tokenized and trimmed.
        :param max_tokens: The maximum number of tokens allowed.
        :return: The trimmed string with no more than max_tokens tokens.
        """
        # Tokenize the input string
        tokens = self.__tokenizer.encode(input_string, return_tensors="pt")[0]

        # Check if the number of tokens exceeds max_tokens
        if len(tokens) > max_tokens:
            # Trim the tokens to max_tokens length
            tokens = tokens[:max_tokens]

        # Decode the tokens back into a string
        trimmed_string = self.__tokenizer.decode(tokens, skip_special_tokens=True)

        return trimmed_string