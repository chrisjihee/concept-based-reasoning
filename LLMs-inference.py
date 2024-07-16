from getpass import getpass

from together import Together
from tqdm import tqdm

from chrisbase.io import *

# setup program
test_size = 3
input_file = "data/LLM-test-with-KG-31.json"
output_file = "data/LLM-test-with-KG-responses-2.json"
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
total_responses = []
test_set = test_set[:test_size]
for i, qa in enumerate(test_set, start=1):
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
    for target_model in tqdm(target_models, desc=f"* Answering question ({i}/{len(test_set)})", unit="model"):
        model_response = chat_with_llm(model_id=target_model,
                                       messages=[{"role": "user", "content": inference_prompt}])
        if model_response:
            model_responses.append({
                "model": target_model,
                "output": model_response,
            })
    qa["responses"] = model_responses
    total_responses.append(qa)
    # save to output file (incremental save)
    save_json(total_responses, output_file, indent=2, ensure_ascii=False)

# write to output file (final save)
save_json(total_responses, output_file, indent=2, ensure_ascii=False)
