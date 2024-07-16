from getpass import getpass

from together import Together
from tqdm import tqdm

from chrisbase.io import *

TOGETHER_API_TOKEN = read_or(first_path_or("together-tokens*")) or getpass()
client = Together(api_key=TOGETHER_API_TOKEN)


def chat_with_llm(messages, model="meta-llama/Llama-3-8b-chat-hf"):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    return ''.join(chunk.choices[0].delta.content or "" for chunk in stream)


# setup arguments
input_file = "data/LLM-test-with-KG-31.json"
output_file = "data/LLM-test-with-KG-responses.json"
target_models = [
    "google/gemma-2b-it",
    "google/gemma-7b-it",
    "togethercomputer/alpaca-7b",
    "lmsys/vicuna-7b-v1.5",
    "lmsys/vicuna-13b-v1.5",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-3-8b-chat-hf",
    "meta-llama/Llama-3-70b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "upstage/SOLAR-10.7B-Instruct-v1.0",
    "Qwen/Qwen1.5-0.5B-Chat",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen1.5-14B-Chat",
    "Qwen/Qwen1.5-32B-Chat",
    "Qwen/Qwen1.5-72B-Chat",
    "Qwen/Qwen1.5-110B-Chat",
    "Qwen/Qwen2-72B-Instruct",
]
prompt_template = read_or("template/inference_prompt.txt") or getpass("Enter the prompt template: ")

# read input file
test_set = load_json(input_file)
demo, test_set = test_set[0], test_set[1:]
demo_question = demo["question"]
demo_answer = demo["answer"]
demo_answer_size = len(demo["answer"].split())
demo_knowledge_size = len(demo["triples"])
demo_triples = "\n".join([f"  - {triple}" for triple in demo["triples"]])

# chat with LLMs
total_responses = []
for i, qa in enumerate(test_set[:2], start=1):
    difficulty = qa["difficulty"]
    real_question = qa["question"]
    real_answer_size = len(qa["answer"].split())
    real_knowledge_size = len(qa["triples"])
    inference_prompt = prompt_template.format(
        real_question=real_question,
        real_answer_size=real_answer_size,
        real_knowledge_size=real_knowledge_size,
        demo_question=demo_question,
        demo_answer=demo_answer,
        demo_answer_size=demo_answer_size,
        demo_knowledge_size=demo_knowledge_size,
        demo_triples=demo_triples,
    )
    model_responses = []
    for model in tqdm(target_models, desc=f"* Answering question ({i}/{len(test_set)})", unit="model"):
        r = {
            "model": model,
            "output": chat_with_llm(model=model,
                                    messages=[{"role": "user", "content": inference_prompt}]),
        }
        model_responses.append(r)
    qa["responses"] = model_responses
    total_responses.append(qa)

# write to output file
save_json(total_responses, output_file)
