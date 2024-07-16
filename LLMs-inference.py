from datetime import datetime
from getpass import getpass

from together import Together
from tqdm import tqdm

from chrisbase.io import *

# setup program
test_size = 3
input_file = "data/LLM-test-with-KG-31.json"
output_file = f"data/LLM-test-with-KG-responses-{test_size}.json"
target_models = [x["full_id"] for x in load_json("conf/full_chat_models.json")]
prompt_template = read_or("template/inference_prompt.txt") or getpass("Enter the prompt template: ")
api_client = Together(timeout=10,
                      max_retries=3,
                      api_key=read_or(first_path_or("together-tokens*")) or getpass("Enter your Together API key: "))


# define function to chat with LLM
def chat_with_llm(messages, model_id):
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
test_set = load_json(input_file)
demo, test_set = test_set[0], test_set[1:]
demo_question = demo["question"]
demo_answer = demo["answer"]
demo_answer_size = len(demo["answer"].split())
demo_knowledge_size = len(demo["triples"])
demo_triples = "\n".join([f"  - {triple}" for triple in demo["triples"]])

# chat with LLMs
total_data = []
test_set = test_set[:test_size]
for i, item in enumerate(test_set, start=1):
    difficulty = item["difficulty"]
    real_question = item["question"]
    real_answer_size = len(item["answer"].split())
    real_knowledge_size = len(item["triples"])
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
    total_data.append(item)
    item["responses"] = []
    for target_model in tqdm(target_models, desc=f"* Answering question ({i}/{len(test_set)})", unit="model"):
        based = datetime.now()
        model_response = chat_with_llm(model_id=target_model,
                                       messages=[{"role": "user", "content": inference_prompt}])
        elasped = datetime.now() - based
        if model_response:
            item["responses"].append({
                "model": target_model,
                "output": model_response,
                "elasped": str(elasped),
            })
        save_json(total_data, output_file, indent=2, ensure_ascii=False)

# write to output file (final save)
save_json(total_data, output_file, indent=2, ensure_ascii=False)
