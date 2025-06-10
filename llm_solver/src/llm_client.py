
import os
from dotenv import load_dotenv
load_dotenv()

from typing import Any, Dict, Type, TypeVar
import json
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class LLMClient:
    def __init__(self):
        # Azure OpenAI configuration
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.api_version = os.getenv("OPENAI_API_VERSION", "")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        
        # Initialize the Azure OpenAI client
        self.client = AzureChatOpenAI(
            azure_endpoint=self.endpoint,
            azure_deployment=self.deployment_name,
            openai_api_version=self.api_version,
            openai_api_key=self.api_key,
            temperature=0.2,
            max_tokens=2000
        )

    def send_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Send a prompt to the Azure OpenAI API via LangChain and get a response.
        
        Args:
            prompt: The text prompt to send to the LLM
            
        Returns:
            Dictionary containing the response
        """
        try:
            # Create messages for the conversation
            messages = [
                SystemMessage(content="You are a helpful assistant specialized in linear programming."),
                HumanMessage(content=prompt)
            ]
            
            # Invoke the model
            response = self.client.invoke(messages)
            
            # Extract content from the response
            content = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "content": content,
                "model": self.deployment_name,
                "usage": {}
            }
        except Exception as e:
            # Log error and return an error message
            print(f"Error calling Azure OpenAI: {str(e)}")
            return {
                "content": f"Error: {str(e)}",
                "model": self.deployment_name,
                "usage": {},
                "error": str(e)
            }
    
    def send_structured_prompt(self, prompt: str, output_class: Type[T]) -> T:
        """
        Send a prompt to the LLM and get a structured response as a Pydantic object.
        
        Args:
            prompt: The text prompt to send to the LLM
            output_class: The Pydantic class to parse the response into
            
        Returns:
            An instance of the specified Pydantic class
        """
        try:
            # Create messages for the conversation
            messages = [
                SystemMessage(content="You are a helpful assistant specialized in linear programming. Return structured data exactly as requested."),
                HumanMessage(content=prompt)
            ]
            
            # Use the function calling approach for structured output
            structured_llm = self.client.with_structured_output(output_class)
            
            # Invoke the model with structured output
            result = structured_llm.invoke(messages)
            
            return result
        except Exception as e:
            # Log error and raise exception
            print(f"Error getting structured output: {str(e)}")
            raise