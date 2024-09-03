from getpass import getpass
from openai import OpenAI

from sklearn.model_selection import train_test_split
from together import Together
from tqdm import tqdm

from chrisbase.data import *
from chrisbase.io import *
from chrisbase.util import *

logger = logging.getLogger(__name__)
args = CommonArguments(
    env=ProjectEnv(
        project="LLM-based",
        job_name="LLM-based-extraction",
        msg_level=logging.INFO,
        msg_format=LoggingFormat.PRINT_00,
    )
)

# setup program
test_size = 100
debug_test_size = 1

request_timeout = 300
default_max_tokens = 512
default_temperature = 0.5
default_repetition_penalty = 1.1
api_client = OpenAI(timeout=request_timeout,
                    api_key=read_or("conf/key-openai.txt") or getpass("OpenAI API key: "))


# define function to chat with GPT
def chat_with_gpt(messages, model_id="gpt-4o"):
    try:
        stream = api_client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=True,
        )
        output = ''.join(chunk.choices[0].delta.content or '' for chunk in stream)
        return output.strip()
    except:
        return None


dataset = "WN18RR" or "YAGO3-10"
prompt_template = read_or("template/extraction_prompt.txt") or getpass("Extraction Prompt: ")
common_prompt = prompt_template
generation_levels = {
    1: "relation_only",  # Relation Classification
    # 2: "tail_only",  # Link Prediction
    # 3: "tail_with_relation",
    # 4: "free_with_quantity",
    # 5: "free_without_quantity",
}
target_generation_levels = sorted(generation_levels.keys())

for generation_level in target_generation_levels:
    input_file = f"generation/{dataset}/edges_as_text_all-responses-{test_size}@{generation_level}.json"
    outut_file = f"extraction/{dataset}/edges_as_text_all-responses-{test_size}@{generation_level}.json"
    test_data = load_json(input_file)
    if debug_test_size > 0:
        test_data = test_data[:debug_test_size]

    # chat with GPT
    with JobTimer(f"KG Extraction(generation_level={generation_level}, num_test={len(test_data)})",
                  rt=1, rb=1, rw=114, rc='=', mt=1, verbose=1):
        output_data = []
        for i, item in enumerate(test_data, start=1):
            output_data.append(item)
            chat_history = []
            entity = item["entity"]
            triples = item["triples"]
            messages = item["messages"]
            model_responses = []
            for j, response in enumerate(item["responses"], start=1):
                assert response["level"] == generation_level, f"level={response['level']} != generation_level={generation_level}"
                model = response["model"].split("/")[-1]
                output = response["output"]
                model_responses.append(f"\n<BEGIN_OF_MODEL_RESPONSE ({j}. {model})>\n{output}\n<END_OF_MODEL_RESPONSE>\n\n")
            custom_prompt = common_prompt.format(
                entity=entity,
                triples_by_human='\n'.join([f'  - {h} -> {r} -> {t}' for (h, r, t) in triples]),
                model_responses="\n".join(model_responses),
                num_model=len(model_responses),
            )
            print(custom_prompt)
