#!/usr/bin/env python3
"""
Verify LLM Setup for Agentic AI Workshop
==========================================

Run this script to verify your LLM configuration is working:
    python verify_setup.py

Supports both DIAL (Azure OpenAI) and OpenAI directly.
"""

import sys
sys.path.insert(0, '.')

from setup_llm import verify_setup, get_chat_model, get_model_name, get_provider_name

def main():
    print("=" * 60)
    print("🔧 Agentic AI Workshop - LLM Setup Verification")
    print("=" * 60)
    print()
    
    # Step 1: Verify environment
    if not verify_setup():
        print("\n⚠️  Please fix the configuration issues above before continuing.")
        print("   1. Copy .env.example to .env")
        print("   2. Set either DIAL_API_KEY or OPENAI_API_KEY")
        sys.exit(1)
    
    # Step 2: Test model connection
    print("\n" + "=" * 60)
    print("🧪 Testing LLM Connection...")
    print("=" * 60)
    
    try:
        model = get_chat_model()
        provider = get_provider_name()
        model_name = get_model_name()
        
        print(f"\n📡 Connecting via {provider} with model: {model_name}")
        
        response = model.invoke("Respond with exactly: 'Connection successful!'")
        print(f"\n✅ Response: {response.content}")
        
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Your LLM setup is working correctly!")
        print("=" * 60)
        print("\nYou can now run the workshop notebooks.")
        
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("   1. Check your API key is correct")
        print("   2. For DIAL: Verify you have access to https://ai-proxy.lab.epam.com")
        print("   3. For OpenAI: Verify your API key at https://platform.openai.com")
        print("   4. Check your network/VPN connection")
        sys.exit(1)


if __name__ == "__main__":
    main()

