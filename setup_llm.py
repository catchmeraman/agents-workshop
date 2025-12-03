"""
LLM Setup Utility for Agentic AI Workshop
=========================================

This module provides helper functions to configure LangChain/LangGraph
with either DIAL (EPAM AI Proxy) or OpenAI directly.

Supported Providers:
    - DIAL (Azure OpenAI via EPAM AI Proxy) - set DIAL_API_KEY
    - OpenAI (Direct) - set OPENAI_API_KEY

Usage in notebooks:
    from setup_llm import get_chat_model, get_embeddings, verify_setup
    
    # Verify setup
    verify_setup()
    
    # Get a chat model (auto-detects provider)
    model = get_chat_model()
    model = get_chat_model(temperature=0.7)
"""

import os
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

# Load environment variables - override=True to override shell env vars
load_dotenv(override=True)


class LLMProvider(Enum):
    """Supported LLM providers."""
    DIAL = "dial"
    OPENAI = "openai"
    UNKNOWN = "unknown"


def detect_provider() -> LLMProvider:
    """
    Auto-detect which LLM provider is configured based on environment variables.
    
    Priority:
    1. DIAL (if DIAL_API_KEY is set)
    2. OpenAI (if OPENAI_API_KEY is set)
    
    Returns:
        LLMProvider enum indicating the detected provider
    """
    dial_key = os.getenv("DIAL_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    # Check DIAL first (prioritize enterprise setup)
    if dial_key and dial_key != "your-dial-api-key-here":
        return LLMProvider.DIAL
    
    # Check OpenAI
    if openai_key and openai_key != "your-openai-api-key-here":
        return LLMProvider.OPENAI
    
    return LLMProvider.UNKNOWN


def verify_setup() -> bool:
    """
    Verify that LLM configuration is properly set up.
    
    Returns:
        True if setup is valid, False otherwise
    """
    print("🔍 Checking LLM Configuration...")
    print("=" * 50)
    
    provider = detect_provider()
    
    if provider == LLMProvider.DIAL:
        print(f"📡 Provider: DIAL (Azure OpenAI via EPAM AI Proxy)")
        return _verify_dial_setup()
    elif provider == LLMProvider.OPENAI:
        print(f"📡 Provider: OpenAI (Direct)")
        return _verify_openai_setup()
    else:
        print("❌ No valid LLM provider configured")
        print("\n📋 To configure, set ONE of the following in your .env file:")
        print("   Option 1 (DIAL): DIAL_API_KEY=your-dial-api-key")
        print("   Option 2 (OpenAI): OPENAI_API_KEY=your-openai-api-key")
        return False


def _verify_dial_setup() -> bool:
    """Verify DIAL configuration."""
    required_vars = [
        ("DIAL_API_KEY", os.getenv("DIAL_API_KEY")),
    ]
    
    optional_vars = [
        ("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT", "https://ai-proxy.lab.epam.com")),
        ("AZURE_OPENAI_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")),
        ("AZURE_OPENAI_DEPLOYMENT_NAME", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")),
    ]
    
    all_set = True
    for var_name, value in required_vars:
        if value and value != "your-dial-api-key-here":
            print(f"✅ {var_name} is set")
        else:
            print(f"❌ {var_name} not set or has placeholder value")
            all_set = False
    
    print("\n📋 Configuration:")
    for var_name, value in optional_vars:
        print(f"   {var_name}: {value}")
    
    if all_set:
        print("\n✅ DIAL setup verified successfully!")
    else:
        print("\n❌ Please update your .env file with valid DIAL credentials")
    
    return all_set


def _verify_openai_setup() -> bool:
    """Verify OpenAI configuration."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    
    if api_key and api_key != "your-openai-api-key-here":
        print(f"✅ OPENAI_API_KEY is set")
        print(f"\n📋 Configuration:")
        print(f"   OPENAI_MODEL_NAME: {model}")
        print("\n✅ OpenAI setup verified successfully!")
        return True
    else:
        print(f"❌ OPENAI_API_KEY not set or has placeholder value")
        return False


def get_chat_model(
    model_name: str = None,
    temperature: float = 0,
    **kwargs
):
    """
    Get a LangChain chat model configured for the detected provider.
    
    Auto-detects whether to use DIAL (Azure OpenAI) or OpenAI based on
    environment variables.
    
    Args:
        model_name: Model/deployment name (default: from env or provider default)
        temperature: Model temperature (default: 0)
        **kwargs: Additional arguments passed to the model
    
    Returns:
        ChatOpenAI or AzureChatOpenAI instance
    
    Examples:
        model = get_chat_model()
        model = get_chat_model("gpt-4", temperature=0.7)
        model = get_chat_model(temperature=0.5, max_tokens=1000)
    """
    provider = detect_provider()
    
    if provider == LLMProvider.DIAL:
        return _get_dial_chat_model(model_name, temperature, **kwargs)
    elif provider == LLMProvider.OPENAI:
        return _get_openai_chat_model(model_name, temperature, **kwargs)
    else:
        raise ValueError(
            "No LLM provider configured. Please set DIAL_API_KEY or OPENAI_API_KEY in your .env file."
        )


def _get_dial_chat_model(deployment_name: str = None, temperature: float = 0, **kwargs):
    """Get a chat model configured for DIAL (Azure OpenAI)."""
    from langchain_openai import AzureChatOpenAI
    
    api_key = os.getenv("DIAL_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://ai-proxy.lab.epam.com")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    
    if deployment_name is None:
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    
    return AzureChatOpenAI(
        azure_deployment=deployment_name,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        temperature=temperature,
        **kwargs
    )


def _get_openai_chat_model(model_name: str = None, temperature: float = 0, **kwargs):
    """Get a chat model configured for OpenAI directly."""
    from langchain_openai import ChatOpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if model_name is None:
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    
    # Explicitly set base_url to override any local OPENAI_BASE_URL env var
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        temperature=temperature,
        **kwargs
    )


def get_embeddings(model_name: str = None, **kwargs):
    """
    Get a LangChain embeddings model configured for the detected provider.
    
    Args:
        model_name: Model/deployment name (default: provider-specific default)
        **kwargs: Additional arguments passed to the embeddings model
    
    Returns:
        OpenAIEmbeddings or AzureOpenAIEmbeddings instance
    
    Example:
        embeddings = get_embeddings()
    """
    provider = detect_provider()
    
    if provider == LLMProvider.DIAL:
        return _get_dial_embeddings(model_name, **kwargs)
    elif provider == LLMProvider.OPENAI:
        return _get_openai_embeddings(model_name, **kwargs)
    else:
        raise ValueError(
            "No LLM provider configured. Please set DIAL_API_KEY or OPENAI_API_KEY in your .env file."
        )


def _get_dial_embeddings(deployment_name: str = None, **kwargs):
    """Get embeddings model configured for DIAL (Azure OpenAI)."""
    from langchain_openai import AzureOpenAIEmbeddings
    
    api_key = os.getenv("DIAL_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://ai-proxy.lab.epam.com")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    
    if deployment_name is None:
        deployment_name = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002")
    
    return AzureOpenAIEmbeddings(
        azure_deployment=deployment_name,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        **kwargs
    )


def _get_openai_embeddings(model_name: str = None, **kwargs):
    """Get embeddings model configured for OpenAI directly."""
    from langchain_openai import OpenAIEmbeddings
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if model_name is None:
        model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
    
    # Explicitly set base_url to override any local OPENAI_BASE_URL env var
    return OpenAIEmbeddings(
        model=model_name,
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        **kwargs
    )


def get_openai_client():
    """
    Get a raw OpenAI client (not LangChain) for direct API access.
    
    Returns:
        OpenAI or AzureOpenAI client instance based on detected provider
    
    Example:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """
    provider = detect_provider()
    
    if provider == LLMProvider.DIAL:
        from openai import AzureOpenAI
        
        api_key = os.getenv("DIAL_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://ai-proxy.lab.epam.com")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        
        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
    elif provider == LLMProvider.OPENAI:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        # Explicitly set base_url to override any local OPENAI_BASE_URL env var
        return OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
    else:
        raise ValueError(
            "No LLM provider configured. Please set DIAL_API_KEY or OPENAI_API_KEY in your .env file."
        )


def get_model_name() -> str:
    """Get the current model/deployment name from environment."""
    provider = detect_provider()
    
    if provider == LLMProvider.DIAL:
        return os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    elif provider == LLMProvider.OPENAI:
        return os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    else:
        return "unknown"


# Backwards compatibility aliases
get_deployment_name = get_model_name


def get_provider_name() -> str:
    """Get a human-readable name of the current provider."""
    provider = detect_provider()
    
    if provider == LLMProvider.DIAL:
        return "DIAL (Azure OpenAI)"
    elif provider == LLMProvider.OPENAI:
        return "OpenAI"
    else:
        return "Not configured"


# Quick test when run directly
if __name__ == "__main__":
    print("\n🚀 LLM Setup Utility Test\n")
    
    if verify_setup():
        print("\n📝 Testing model initialization...")
        try:
            model = get_chat_model()
            print(f"✅ Model created: {get_model_name()} via {get_provider_name()}")
            
            # Quick test
            response = model.invoke("Say 'Hello' in one word")
            print(f"✅ Test response: {response.content}")
        except Exception as e:
            print(f"❌ Model test failed: {e}")
