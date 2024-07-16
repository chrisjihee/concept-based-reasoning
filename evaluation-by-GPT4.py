from getpass import getpass

from openai import OpenAI
from tqdm import tqdm

from chrisbase.io import *

# setup arguments
test_size = 1
request_timeout = 300
input_file = "data/LLM-test-with-KG-responses-3.json"
prompt_template = read_or("template/extraction_prompt.txt") or getpass("Extraction Prompt: ")
api_client = OpenAI(timeout=request_timeout,
                    api_key=read_or("conf/key-openai.txt") or getpass("OpenAI API key: "))


# define function to chat with LLM
def chat_with_gpt(messages, model_id="gpt-4o"):
    try:
        stream = api_client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=True,
        )
        return ''.join(chunk.choices[0].delta.content or "" for chunk in stream)
    except:
        return None


# read input file
input_data = load_json(input_file)
input_data = input_data[:test_size]
for i, item in enumerate(input_data, start=1):
    chat_history = []
    question = item["question"]
    base_answer = item["answer"]
    base_triples = "\n".join([f"  - {triple}" for triple in item["triples"]])
    model_responses = []
    for response in item["responses"]:
        model = response["model"].split("/")[-1]
        output = response["output"]
        model_responses.append(f"\n----------\n[{model}]:\n{output}\n----------\n\n")
    extraction_prompt = prompt_template.format(
        question=question,
        base_answer=base_answer,
        base_triples=base_triples,
        model_responses="\n".join(model_responses),
        num_model=len(model_responses),
    )
    chat_history.append({"role": "user", "content": extraction_prompt})
    print("=" * 120)
    print(extraction_prompt)
    print("=" * 120)
    print("\n" * 5)
    model_response = chat_with_gpt(model_id="gpt-4o", messages=chat_history)
    print("=" * 120)
    print(response)
    print("=" * 120)
