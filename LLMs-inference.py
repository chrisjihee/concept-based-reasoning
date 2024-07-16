from getpass import getpass

from together import Together

from chrisbase.io import *
from chrisbase.util import *

TOGETHER_API_TOKEN = read_or(first_path_or("together-tokens*")) or getpass()
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_TOKEN
print(f'TOGETHER_API_KEY = {mask_str(os.environ.get("TOGETHER_API_KEY"), start=4, end=-4)}')

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))


def chat_with_llm(messages, model="meta-llama/Llama-3-8b-chat-hf"):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    return ''.join(chunk.choices[0].delta.content or "" for chunk in stream)


single_prompt = """
The typical color of a llama is what?
Answer in one word.
"""
prompt_history = [
    {"role": "user", "content": single_prompt}
]

response = chat_with_llm(prompt_history, model="meta-llama/Llama-3-8b-chat-hf")
print(response)
