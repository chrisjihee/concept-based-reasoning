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
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = read_or("conf/key-openai.txt") or getpass("OpenAI API key: ")
if "TOGETHER_API_KEY" not in os.environ:
    os.environ["TOGETHER_API_KEY"] = read_or("conf/key-togetherai.txt") or getpass("TogetherAI API key: ")

# setup program
test_size = 100
debug_test_size = 1

openai_assistants = [
    "gpt-4o-2024-08-06",
    # "gpt-4o-mini"
]
togethers_assistants = [
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    # "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
]
openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
together_client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))


# define function to chat with LLM through OpenAI
def chat_with_llm_by_OpenAI(**kwargs):
    try:
        response = openai_client.chat.completions.create(**kwargs)
        output = response.choices[0].message.content
        return output
    except Exception as e:
        logger.error("Exception:", e)
        return None


# define function to chat with LLM through TogetherAI
def chat_with_llm_by_Together(**kwargs):
    try:
        response = together_client.chat.completions.create(**kwargs)
        output = response.choices[0].message.content
        return output
    except Exception as e:
        logger.error("Exception:", e)
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
            entity = item["entity"]
            triples = item["triples"]
            messages = item["messages"]
            model_responses = []
            for j, response in enumerate(item["responses"], start=1):
                assert response["level"] == generation_level, f"level={response['level']} != generation_level={generation_level}"
                model = response["model"].split("/")[-1]
                output = response["output"]
                model_responses.append(f"\n<BEGIN_OF_MODEL_RESPONSE ({j}. {model})>\n{output}\n<END_OF_MODEL_RESPONSE ({j}. {model})>\n\n")
            custom_prompt = common_prompt.format(
                entity=entity,
                triples_by_human='\n'.join([f'  - {h} -> {r} -> {t}' for (h, r, t) in triples]),
                model_responses="\n".join(model_responses),
                num_model=len(model_responses),
            )
            chat_history = [
                {
                    "role": "system",
                    "content": "You will be provided with unstructured data, and your task is to parse it into JSON format."
                },
                {
                    "role": "user",
                    "content": custom_prompt
                }
            ]
            output_data.append({
                "entity": entity,
                "triples": triples,
            })
            print(custom_prompt)
            for teacher in openai_assistants:
                model_output = chat_with_llm_by_OpenAI(messages=chat_history, model=teacher, max_tokens=2048)
                print(f"=== GPT [{teacher}] ===")
                print(model_output)
                print("=" * 80)
                print("\n\n")
            for teacher in togethers_assistants:
                model_output = chat_with_llm_by_Together(messages=chat_history, model=teacher, max_tokens=2048)
                print(f"=== LLaMA [{teacher}] ===")
                print(model_output)
                print("=" * 80)
                print("\n\n")
