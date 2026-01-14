import os
import sys
from ollama import Client

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Ollama Cloud API test
api_key = os.environ.get('OLLAMA_API_KEY')
print(f"API Key present: {bool(api_key)}")
print(f"API Key length: {len(api_key) if api_key else 0}")

# Create client for Ollama Cloud (correct endpoint from docs)
client = Client(
    host='https://ollama.com',
    headers={'Authorization': f'Bearer {api_key}'}
)

# Test cloud-exclusive models (not available in local server)
cloud_models = [
    #('glm-4.6', 'Cloud-exclusive GLM model'),
    #('qwen3-coder:480b-cloud', 'Massive 480B coding model'),
    ('gpt-oss:120b-cloud', 'Large 120B GPT-OSS'),
    #('deepseek-v3.1:671b-cloud', 'Massive 671B DeepSeek model'),
    #('minimax-m2:cloud', 'MiniMax M2 - free on Ollama cloud until Nov 7'),
]

for model, description in cloud_models:
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"Description: {description}")
    print('='*60)
    
    try:
        response = client.chat(
            model=model,
            messages=[
                {'role': 'user', 'content': 'Why are rainbow colors specifically 7? Answer briefly.'}
            ],
            stream=True
        )
        
        for chunk in response:
            print(chunk['message']['content'], end='', flush=True)
        print()  # Final newline
        print("✓ Model completed successfully")
        
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        break
    except Exception as e:
        print(f"✗ Error: {e}")
    
print(f"\n{'='*60}")
print("Ollama Cloud test complete!")
print('='*60)