from getpass import getpass

from tqdm import tqdm

from chrisbase.io import *

# setup arguments
input_file = "data/LLM-test-with-KG-responses-3.json"
output_file = "data/LLM-test-with-KG-extraction-prompt-{i}.txt"
prompt_template = read_or("template/extraction_prompt.txt") or getpass("Enter the prompt template: ")

total_data = load_json(input_file)
for i, item in enumerate(tqdm(total_data, desc=f"* Making extraction prompt", unit="item"), start=1):
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
        num_model=len(model_responses),
        model_responses="\n".join(model_responses),
    )
    write_or(output_file.format(i=i), extraction_prompt)
