from getpass import getpass

from chrisbase.data import *
from chrisbase.io import *

logger = logging.getLogger(__name__)
args = CommonArguments(
    env=ProjectEnv(
        project="LLM-based",
        job_name="LLM-based-generation",
        msg_level=logging.INFO,
        msg_format=LoggingFormat.PRINT_00,
    )
)

# setup arguments
test_size = 30
input_file = "data/LLM-test-with-KG-responses-30.json"
extraction_template = read_or("template/extraction_prompt.txt") or getpass("Extraction Prompt: ")


def limit_words(text, max_words):
    text = text.replace("\n", "<BR> </BR>")
    text = ' '.join(text.split(' ')[:max_words])
    text = text.replace("<BR> </BR>", "\n")
    return text


# read input file
input_data = load_json(input_file)

# chat with GPT
with JobTimer("Prompt Generation", rt=1, rb=1, rw=114, rc='=', verbose=1):
    total_data = []
    input_data = input_data[:test_size]
    for i, item in enumerate(input_data, start=1):
        total_data.append(item)
        question = item["question"]
        base_answer = item["answer"]
        base_triples = "\n".join([f"  - {triple}" for triple in item["triples"]])
        max_output_words = int(item["avg_words"] * 1.5)
        model_responses = []
        for j, response in enumerate(item["responses"], start=1):
            model = response["model"].split("/")[-1]
            output = limit_words(response["output"], max_words=max_output_words)
            model_responses.append(f"\n<BEGIN_OF_MODEL_RESPONSE ({j}. {model})>\n{output}\n<END_OF_MODEL_RESPONSE>\n\n")

        extraction_prompt = extraction_template.format(
            question=question,
            base_answer=base_answer,
            base_triples=base_triples,
            model_responses="\n".join(model_responses),
            num_model=len(model_responses),
        )
        write_or("data/LLM-test-with-KG-extraction-prompt-{i}.txt".format(i=i), extraction_prompt)
