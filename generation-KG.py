from getpass import getpass

from openai import OpenAI
from sklearn.model_selection import train_test_split
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
        job_name="generation-KG",
        message_level=logging.INFO,
        message_format=LoggingFormat.BRIEF_00,
    )
)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = read_or("conf/key-openai-GPT4.txt") or getpass("OpenAI API key: ")
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
            "content": choice.message.content.strip(),
            "finish_reason": choice.finish_reason,
        }
    except Exception as e:
        return {
            "role": "report",
            "content": str(e),
            "finish_reason": f"{type(e).__name__}",
        }


# define function to chat with LLM through TogetherAI
def chat_with_LLM_by_Together(**kwargs):
    try:
        response = together_client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        return {
            "role": choice.message.role.value,
            "content": choice.message.content.strip(),
            "finish_reason": choice.finish_reason.value,
        }
    except Exception as e:
        return {
            "role": "report",
            "content": str(e),
            "finish_reason": f"{type(e).__name__}",
        }


# define function to normalize simple list in json
def normalize_simple_list_in_json(json_input):
    json_output = []
    pattern = re.compile(r"\[[^\[\]]+?]")
    if re.search(pattern, json_input):
        pre_end = 0
        for m in re.finditer(pattern, json_input):
            json_output.append(m.string[pre_end: m.start()])
            json_output.append("[" + " ".join(m.group().split()).removeprefix("[ ").removesuffix(" ]") + "]")
            pre_end = m.end()
        json_output.append(m.string[pre_end:])
        return ''.join(json_output)
    else:
        return json_input


# setup program
test_size = 100
debug_test_size = -1
max_entity_triples = 7
num_demo_group = 10
demo_groups = [1, 3, 5]
each_demo_group_size = 1
dataset_names = [
    # "WN18RR",
    "YAGO3-10",
]
generation_levels = {
    1: "relation_only",  # Relation Classification
    2: "tail_only",  # Link Prediction
    3: "tail_with_relation",
    # 4: "free_with_quantity",
    # 5: "free_without_quantity",
}
generation_models = [
    # ("mistralai/Mistral-7B-Instruct-v0.2", "text"),
    # ("mistralai/Mixtral-8x7B-Instruct-v0.1", "text"),
    # ("mistralai/Mixtral-8x7B-Instruct-v0.1", "json"),
    # ("mistralai/Mixtral-8x22B-Instruct-v0.1", "text"),
    # ("meta-llama/Meta-Llama-3-8B-Instruct-Turbo", "text"),
    # ("meta-llama/Meta-Llama-3-70B-Instruct-Turbo", "text"),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "text"),
    # ("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "json"),
    # ("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "text"),
    # ("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "json"),
    # ("meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", "text"),
    # ("gpt-4o-mini-2024-07-18", "text"),
    # ("gpt-4o-mini-2024-07-18", "json"),
    # ("gpt-4o-2024-08-06", "text"),
    # ("gpt-4o-2024-08-06", "json"),
]
max_tokens = 4000
system_prompt = "You will be provided with an target entity and demo examples, and your task is to generate knowledge triples. Respond in JSON format without any other explanation."
generation_prompt = read_or("template/generation-KG.txt") or getpass("Generation KG Prompt: ")
random_seed = 70

# run program
for dataset_name in dataset_names:
    dataset_file = f"data/{dataset_name}/edges_as_text_all.tsv"
    total_triples = list(tsv_lines(dataset_file))
    total_entities = [{"entity": k, "triples": sorted(v)} for k, v in grouped(total_triples, key=lambda x: x[0])]
    total_entities = [x for x in total_entities if len(x["triples"]) <= max_entity_triples]
    defined_relations = sorted({x[1] for x in total_triples})  # TODO: relation의 분포 비율까지 힌트로 줄까?
    train_data, test_data = train_test_split(total_entities, test_size=test_size, random_state=random_seed)
    if debug_test_size > 0:
        test_data = test_data[:debug_test_size]

    test_data_per_size = {k: list(v) for k, v in grouped(test_data, key=lambda x: len(x["triples"]))}
    train_data_per_size = {k: list(v) for k, v in grouped(train_data, key=lambda x: len(x["triples"]))}
    print(f"dataset_file: {dataset_file}")
    print(f"total_triples: {len(total_triples)}")
    print(f"total_entities: {len(total_entities)}")
    print(f"defined_relations: {defined_relations}")
    print(f"train_data: {len(train_data)}")
    print(f"test_data: {len(test_data)}")
    print("test_data_per_size:", {k: len(v) for k, v in test_data_per_size.items()})
    print("train_data_per_size:", {k: len(v) for k, v in train_data_per_size.items()})
    demo_examples = []
    for size in demo_groups:
        for sample in shuffled(test_data_per_size[size], seed=random_seed)[:each_demo_group_size]:
            demo_examples.append(normalize_simple_list_in_json(json.dumps(
                {
                    "entity": sample["entity"],
                    "triples": sample["triples"],
                }, indent=2, ensure_ascii=False,
            )))

    for generation_level in sorted(generation_levels.keys()):
        # generation_file = f"generation/{dataset_name}/edges_as_text_all-responses-{test_size}@{generation_level}.json"
        generation_file = f"generation/{dataset_name}/edges_as_text_all-messages-{len(train_data)}@{generation_level}.jsonl"
        generation_data = []

        with JobTimer(f"KG Generation(dataset_name={dataset_name}, defined_relations={len(defined_relations)}, generation_level={generation_level}, num_train={len(train_data)}, generation_models={len(generation_models)}, max_tokens={max_tokens})",
                      rt=1, rb=1, rw=114, rc='=', mt=1, verbose=1), Path(generation_file).open("w") as fout:
            for i, sample in enumerate(train_data, start=1):
                target_entity = sample["entity"]
                triples_by_human = sample["triples"]
                number_of_triples = len(triples_by_human)
                triples_by_model = []
                if generation_level == 1:
                    for (h, r, t) in triples_by_human:
                        triples_by_model.append((h, "(predicted_relation)", t))
                elif generation_level == 2:
                    for (h, r, t) in triples_by_human:
                        triples_by_model.append((h, r, "(predicted_entity)"))
                elif generation_level == 3:
                    for (h, r, t) in triples_by_human:
                        triples_by_model.append((h, "(predicted_relation)", "(predicted_entity)"))
                elif generation_level == 4:
                    triples_by_model.append((h, "(predicted_relation)", "(predicted_entity)"))
                    triples_by_model.append("...")
                elif generation_level == 5:
                    triples_by_model.append((h, "(predicted_relation)", "(predicted_entity)"))
                    triples_by_model.append("...")
                    number_of_triples = "unknown"
                else:
                    assert False, f"Invalid generation_level: {generation_level}"
                actual_generation_prompt = generation_prompt.format(
                    defined_relations=json.dumps(defined_relations, indent=2),
                    generation_demo_examples="\n\n".join(f"<demo>\n{x}\n</demo>" for x in demo_examples),
                    generation_form=normalize_simple_list_in_json(json.dumps(
                        {
                            "target_entity": target_entity,
                            "triples_by_model": triples_by_model,
                            "number_of_triples": number_of_triples,
                        }, indent=2, ensure_ascii=False,
                    )),
                )
                generation_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": actual_generation_prompt},
                ]
                generation_result = {
                    "dataset_name": dataset_name,
                    "entity": target_entity,
                    "triples_by_human": triples_by_human,
                    "generation_level": generation_level,
                    "generation_messages": generation_messages,
                    # "generation_outputs": [],
                    # "generation_errors": [],
                }
                fout.write(json.dumps(generation_result, ensure_ascii=False) + "\n")
                generation_data.append(generation_result)
                # print("\n" * 3)
                # print(f'<triples_by_human>\n{normalize_simple_list_in_json(json.dumps(triples_by_human, indent=2, ensure_ascii=False))}\n</triples_by_human>')
                # for (generation_model, generation_type) in tqdm(generation_models, desc=f"* Generating KG ({i}/{len(test_data)})", unit="model", file=sys.stdout):
                #     based = datetime.now()
                #     if generation_model.startswith("gpt-"):
                #         if generation_type == "json":
                #             generation_output = chat_with_LLM_by_OpenAI(
                #                 model=generation_model,
                #                 messages=generation_messages,
                #                 max_tokens=max_tokens,
                #                 response_format={"type": "json_object"},
                #             )
                #         else:
                #             generation_output = chat_with_LLM_by_OpenAI(
                #                 model=generation_model,
                #                 messages=generation_messages,
                #                 max_tokens=max_tokens,
                #             )
                #     else:
                #         if generation_type == "json":
                #             generation_output = chat_with_LLM_by_Together(
                #                 model=generation_model,
                #                 messages=generation_messages,
                #                 max_tokens=max_tokens,
                #                 response_format={"type": "json_object"},
                #             )
                #         else:
                #             generation_output = chat_with_LLM_by_Together(
                #                 model=generation_model,
                #                 messages=generation_messages,
                #                 max_tokens=max_tokens,
                #             )
                #     generation_seconds = (datetime.now() - based).total_seconds()
                #     if generation_output and generation_output["content"]:
                #         content_len = len(str(generation_output["content"]))
                #         generation_result["generation_outputs"].append({
                #             "type": generation_type,
                #             "model": generation_model,
                #             "output": generation_output,
                #             "seconds": generation_seconds,
                #             "content_len": content_len,
                #         })
                #     else:
                #         generation_result["generation_errors"].append({
                #             "type": generation_type,
                #             "model": generation_model,
                #             "output": generation_output,
                #             "seconds": generation_seconds,
                #         })
                #     # print("\n" * 3)
                #     # print("=" * 200)
                #     # print(f'<generation_type>{generation_type}</generation_type>')
                #     # print(f'<generation_model>{generation_model}</generation_model>')
                #     # if generation_output and "content" in generation_output:
                #     #     print(f'<generation_output_content>\n{generation_output["content"]}\n</generation_output_content>')
                #     # print("=" * 200)
                #     save_json(generation_data, generation_file, indent=2, ensure_ascii=False)

        # save_json(generation_data, generation_file, indent=2, ensure_ascii=False)
