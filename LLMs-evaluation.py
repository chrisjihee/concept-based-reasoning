from getpass import getpass

from together import Together
from tqdm import tqdm

from chrisbase.io import *

# setup arguments
input_file = "data/LLM-test-with-KG-responses.json"
output_file = "data/LLM-test-with-KG-extraction-prompt-{i}.txt"
prompt_template = read_or("template/extraction_prompt.txt") or getpass("Enter the prompt template: ")

total_responses = load_json(input_file)
for i, qa_response in enumerate(total_responses, start=1):
    question = qa_response["question"]
    base_answer = qa_response["answer"]
    base_triples = "\n".join([f"  - {triple}" for triple in qa_response["triples"]])
    model_responses = []
    for response in qa_response["responses"]:
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
