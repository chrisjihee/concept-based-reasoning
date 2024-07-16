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
test_size = 3
input_file = "data/LLM-test-with-KG-31.json"
output_file = "data/LLM-test-with-KG-responses.json"
target_models = [x["full_id"] for x in load_json("conf/full_chat_models.json")]
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
for i, qa in enumerate(test_set[:test_size], start=1):
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
