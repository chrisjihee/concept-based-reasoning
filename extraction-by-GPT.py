from getpass import getpass

from openai import OpenAI
from together import Together
from tqdm import tqdm

from chrisbase.data import *
from chrisbase.io import *
from chrisbase.util import *

# setup environment
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)
args = CommonArguments(
    env=ProjectEnv(
        project="LLM-based",
        job_name="LLM-based-extraction",
        msg_level=logging.INFO,
        msg_format=LoggingFormat.BRIEF_00,
    )
)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = read_or("conf/key-openai.txt") or getpass("OpenAI API key: ")
if "TOGETHER_API_KEY" not in os.environ:
    os.environ["TOGETHER_API_KEY"] = read_or("conf/key-togetherai.txt") or getpass("TogetherAI API key: ")


# define function to chat with LLM through OpenAI
openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

def chat_with_LLM_by_OpenAI(**kwargs):
    try:
        response = openai_client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        return {
            "role": choice.message.role,
            "content": choice.message.content,
            "finish_reason": choice.finish_reason,
        }
    except Exception as e:
        logger.error("Exception:", e)
        return None


# define function to chat with LLM through TogetherAI
together_client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

def chat_with_LLM_by_Together(**kwargs):
    try:
        response = together_client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        return {
            "role": choice.message.role.value,
            "content": choice.message.content,
            "finish_reason": choice.finish_reason.value,
        }
    except Exception as e:
        logger.error("Exception:", e)
        return None


# setup program
test_size = 100
debug_test_size = -1
dataset_names = [
    "WN18RR",
    "YAGO3-10",
]
generation_levels = {
    1: "relation_only",  # Relation Classification
    # 2: "tail_only",  # Link Prediction
    # 3: "tail_with_relation",
    # 4: "free_with_quantity",
    # 5: "free_without_quantity",
}
target_generation_levels = sorted(generation_levels.keys())
target_models = [
    "gpt-4o-2024-08-06",
    # "gpt-4o-mini"
    # "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    # "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
]
prompt_template = read_or("template/extraction_prompt.txt") or getpass("Extraction Prompt: ")
common_prompt = prompt_template

# run program
for dataset_name in dataset_names:
    for generation_level in target_generation_levels:
        generation_file = f"generation/{dataset_name}/edges_as_text_all-responses-{test_size}@{generation_level}.json"
        extraction_file = f"extraction/{dataset_name}/edges_as_text_all-responses-{test_size}@{generation_level}.json"

        generation_data = load_json(generation_file)
        if debug_test_size > 0:
            generation_data = generation_data[:debug_test_size]
        max_tokens = 4000

        with JobTimer(f"KG Extraction(dataset_name={dataset_name}, generation_level={generation_level}, num_generation={len(generation_data)}, max_tokens={max_tokens})",
                      rt=1, rb=1, rw=114, rc='=', mt=1, verbose=1):
            output_data = []
            for i, item in enumerate(tqdm(generation_data, desc=f"* Extracting KG", unit="item", file=sys.stdout), start=1):
                entity = item["entity"]
                triples = item["triples"]
                generation_messages = item["messages"]
                model_responses = []
                for j, response in enumerate(item["responses"], start=1):
                    assert response["level"] == generation_level, f"level={response['level']} != generation_level={generation_level}"
                    model = response["model"].split("/")[-1]
                    output = response["output"]
                    model_responses.append(f"\n<BEGIN_OF_MODEL_RESPONSE ({j}. {model})>\n{output}\n<END_OF_MODEL_RESPONSE ({j}. {model})>\n\n")
                extraction_prompt = common_prompt.format(
                    entity=entity,
                    triples_by_human='\n'.join([f'  - {h} -> {r} -> {t}' for (h, r, t) in triples]),
                    model_responses="\n".join(model_responses),
                    num_model=len(model_responses),
                )
                extraction_messages = [
                    {
                        "role": "system",
                        "content": "You will be provided with unstructured data, and your task is to parse it into JSON format."
                    },
                    {
                        "role": "user",
                        "content": extraction_prompt
                    }
                ]
                output_item = {
                    "entity": entity,
                    "triples": triples,
                    "generation_messages": generation_messages,
                    "extraction_messages": extraction_messages,
                    "responses": [],
                    "no_responses": [],
                }
                output_data.append(output_item)
                for model in target_models:
                    based = datetime.now()
                    if model.startswith("gpt-"):
                        output = chat_with_LLM_by_OpenAI(messages=extraction_messages, model=model, max_tokens=max_tokens)
                    else:
                        output = chat_with_LLM_by_Together(messages=extraction_messages, model=model, max_tokens=max_tokens)
                    seconds = (datetime.now() - based).total_seconds()
                    if output and output["content"]:
                        num_chars = len(str(output["content"]))
                        num_words = len(str(output["content"]).split())
                        output_item["responses"].append({
                            "model": model,
                            "output": output,
                            "words": num_words,
                            "chars": num_chars,
                            "seconds": seconds,
                        })
                    else:
                        output_item["no_responses"].append(model)
                save_json(output_data, extraction_file, indent=2, ensure_ascii=False)
            save_json(output_data, extraction_file, indent=2, ensure_ascii=False)
