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
openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
if "TOGETHER_API_KEY" not in os.environ:
    os.environ["TOGETHER_API_KEY"] = read_or("conf/key-togetherai.txt") or getpass("TogetherAI API key: ")
together_client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))


# define function to chat with LLM through OpenAI
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
    # 1: "relation_only",  # Relation Classification
    2: "tail_only",  # Link Prediction
    3: "tail_with_relation",
    4: "free_with_quantity",
    5: "free_without_quantity",
}
extraction_models = [
    "gpt-4o-2024-08-06",
    # "gpt-4o-mini"
    # "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    # "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
]
max_tokens = 4000
system_prompt = "You will be provided with unstructured data, and your task is to parse it into JSON format."
extraction_prompt = read_or("template/extraction_KGs.txt") or getpass("Extraction KGs Prompt: ")

# run program
for dataset_name in dataset_names:
    for generation_level in sorted(generation_levels.keys()):
        generation_file = f"generation/{dataset_name}/edges_as_text_all-responses-{test_size}@{generation_level}.json"
        extraction_file = f"extraction/{dataset_name}/edges_as_text_all-responses-{test_size}@{generation_level}.json"

        generation_data = load_json(generation_file)
        if debug_test_size > 0:
            generation_data = generation_data[:debug_test_size]
        extraction_data = []

        with JobTimer(f"KG Extraction(dataset_name={dataset_name}, generation_level={generation_level}, num_generation={len(generation_data)}, extraction_models={extraction_models}, max_tokens={max_tokens})",
                      rt=1, rb=1, rw=114, rc='=', mt=1, verbose=1):
            for i, sample in enumerate(tqdm(generation_data, desc=f"* Extracting KG", unit="sample", file=sys.stdout), start=1):
                entity = sample["entity"]
                triples_by_human = sample["triples"]
                generation_messages = sample["messages"]
                model_responses = []
                for j, response in enumerate(sample["responses"], start=1):
                    assert response["level"] == generation_level, f"level={response['level']} != generation_level={generation_level}"
                    model = response["model"].split("/")[-1]
                    output = response["output"]
                    model_responses.append(f"\n<BEGIN_OF_MODEL_RESPONSE ({j}. {model})>\n{output}\n<END_OF_MODEL_RESPONSE ({j}. {model})>\n\n")
                actual_extraction_prompt = extraction_prompt.format(
                    entity=entity,
                    triples_by_human='\n'.join([f'  - {h} -> {r} -> {t}' for (h, r, t) in triples_by_human]),
                    model_responses="\n".join(model_responses),
                    num_model=len(model_responses),
                )
                extraction_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": actual_extraction_prompt},
                ]
                extraction_result = {
                    "entity": entity,
                    "triples": triples_by_human,
                    "generation_messages": generation_messages,
                    "extraction_messages": extraction_messages,
                    "extraction_outputs": [],
                    "extraction_errors": [],
                }
                extraction_data.append(extraction_result)
                for extraction_model in extraction_models:
                    based = datetime.now()
                    if extraction_model.startswith("gpt-"):
                        extraction_output = chat_with_LLM_by_OpenAI(messages=extraction_messages, model=extraction_model, max_tokens=max_tokens)
                    else:
                        extraction_output = chat_with_LLM_by_Together(messages=extraction_messages, model=extraction_model, max_tokens=max_tokens)
                    seconds = (datetime.now() - based).total_seconds()
                    if extraction_output and extraction_output["content"]:
                        content_len = len(str(extraction_output["content"]))
                        extraction_result["extraction_outputs"].append({
                            "model": extraction_model,
                            "output": extraction_output,
                            "seconds": seconds,
                            "content_len": content_len,
                        })
                    else:
                        extraction_result["extraction_errors"].append({
                            "model": extraction_model,
                            "output": extraction_output,
                            "seconds": seconds,
                        })
                save_json(extraction_data, extraction_file, indent=2, ensure_ascii=False)

        save_json(extraction_data, extraction_file, indent=2, ensure_ascii=False)
