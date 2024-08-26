import re
from getpass import getpass

from sklearn.model_selection import train_test_split
from together import Together
from tqdm import tqdm

from chrisbase.data import *
from chrisbase.io import *
from chrisbase.util import grouped

logger = logging.getLogger(__name__)
args = CommonArguments(
    env=ProjectEnv(
        project="LLM-based",
        job_name="LLM-based-generation",
        msg_level=logging.INFO,
        msg_format=LoggingFormat.PRINT_00,
    )
)

# setup program
test_size = 100
debug_test_size = 1
demo_size_per_size = 1

request_timeout = 60
default_max_tokens = 512
default_temperature = 0.5
default_repetition_penalty = 1.1
api_client = Together(timeout=request_timeout, max_retries=1,
                      api_key=read_or("conf/key-togetherai.txt") or getpass("TogetherAI API key: "))


# define function to chat with LLM
def chat_with_llm(messages, model_id,
                  max_tokens=default_max_tokens,
                  temperature=default_temperature,
                  repetition_penalty=default_repetition_penalty):
    try:
        stream = api_client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        output = ''.join(chunk.choices[0].delta.content or '' for chunk in stream)
        output = '\n'.join(x.rstrip() for x in output.split('\n'))
        output = re.sub(r"\n\n+", "\n\n", output)
        return output.strip()
    except:
        return None


dataset = "WN18RR"
input_file = f"data/{dataset}/edges_as_text_all.tsv"
output_file = f"data/{dataset}/edges_as_text_all-responses-{test_size}.json"
target_models = [x["full_id"] for x in load_json("conf/core_chat_models.json")]
prompt_template = read_or("template/generation_prompt.txt") or getpass("Generation Prompt: ")
demo_template_match = re.search(r'<BEGIN_OF_DEMO_EXAMPLE>(?s:.)+<END_OF_DEMO_EXAMPLE>', prompt_template)
demo_template = prompt_template[demo_template_match.start(): demo_template_match.end()]

all_samples = list(tsv_lines(input_file))
relations = sorted({x[1] for x in all_samples})
# relations = '\n'.join(f'- {a}' for a in sorted({x[1] for x in all_samples}))

total_data = [{"entity": k, "triples": list(v)} for k, v in grouped(all_samples, key=lambda x: x[0])]
train_data, test_data = train_test_split(total_data, test_size=test_size, random_state=7)
print(f"- #relations: {len(relations)}")
print(f"- #test entities: {len(test_data)}")
print(f"- #train entities: {len(train_data)}")

test_data_per_size = {k: list(v) for k, v in grouped(test_data, key=lambda x: len(x["triples"]))}
train_data_per_size = {k: list(v) for k, v in grouped(train_data, key=lambda x: len(x["triples"]))}
demo_data = []
for s in test_data_per_size.keys():
    if s in train_data_per_size:
        demo_data += train_data_per_size[s][:demo_size_per_size]
demo_prompts = [
    demo_template.format(
        demo_entity=demo["entity"],
        demo_triples_size=len(demo["triples"]),
        demo_triples='\n'.join([f'  - {h} -> {r} -> {t}' for (h, r, t) in demo["triples"]]),
    ) for demo in demo_data
]
generation_prompt = prompt_template[:demo_template_match.start()].format(
    relations='\n'.join(f'- {a}' for a in sorted({x[1] for x in all_samples}))
) + "\n\n".join(demo_prompts) + prompt_template[demo_template_match.end():]
generation_levels = {
    1: "relation_only",
    2: "tail_only",
    3: "tail_with_relation",
    4: "free_form",
}
generation_level = 4
print(f"- generation_level: {generation_level}")

# chat with LLMs
with JobTimer("KG Generation", rt=1, rb=1, rw=114, rc='=', mt=1, verbose=1):
    total_data = []
    if debug_test_size > 0:
        test_data = test_data[:debug_test_size]
    for i, item in enumerate(test_data, start=1):
        total_data.append(item)
        chat_history = []
        test_entity = item["entity"]
        test_triples = item["triples"]
        test_triples_size = len(test_triples)
        if generation_level == 1:
            output_triples = '\n'.join([f'  - {h} -> (predict relation here) -> {t}' for (h, r, t) in test_triples])
        elif generation_level == 2:
            output_triples = '\n'.join([f'  - {h} -> {r} -> (predict entity here)' for (h, r, t) in test_triples])
        elif generation_level == 3:
            output_triples = '\n'.join([f'  - {h} -> (predict relation here) -> (predict entity here)' for (h, r, t) in test_triples])
        elif generation_level == 4:
            output_triples = "(predict triples here)"
        else:
            assert False, f"Invalid generation_level: {generation_level}"
        generation_prompt = generation_prompt.format(
            test_entity=test_entity,
            test_triples_size=test_triples_size,
            output_triples=output_triples,
        )
        # print(f"generation_prompt2={generation_prompt}")
        # exit(1)
        chat_history.append({"role": "user", "content": generation_prompt})

        tot_words = []
        tot_chars = []
        tot_seconds = []
        item["tot_words"] = tot_words
        item["tot_chars"] = tot_chars
        item["tot_seconds"] = tot_seconds
        item["avg_words"] = 0.0
        item["avg_chars"] = 0.0
        item["avg_seconds"] = 0.0
        item["responses"] = []
        item["no_responses"] = []
        for target_model in tqdm(target_models, desc=f"* Constructing KG ({i}/{len(test_data)})", unit="model", file=sys.stdout):
            based = datetime.now()
            model_output = chat_with_llm(messages=chat_history, model_id=target_model)
            seconds = (datetime.now() - based).total_seconds()
            if model_output:
                num_chars = len(model_output)
                num_words = len(model_output.split())
                item["responses"].append({
                    "model": target_model,
                    "output": model_output,
                    "words": num_words,
                    "chars": num_chars,
                    "seconds": seconds,
                })
                tot_words.append(num_words)
                tot_chars.append(num_chars)
                tot_seconds.append(seconds)
                item["avg_words"] = sum(tot_words) / len(tot_words)
                item["avg_chars"] = sum(tot_chars) / len(tot_chars)
                item["avg_seconds"] = sum(tot_seconds) / len(tot_seconds)
            else:
                item["no_responses"].append(target_model)
            save_json(total_data, output_file, indent=2, ensure_ascii=False)

            print("\n" * 3)
            print(f"model = \n{target_model}")
            print("\n" * 3)
            print(f"model_output = \n{model_output}")
            print("\n" * 3)
            aaa = '\n'.join([f'  - {h} -> {r} -> {t}' for (h, r, t) in test_triples])
            print(f"test_triples = \n{aaa}")
            print("\n" * 3)
            print("=" * 120)
            print("\n" * 3)

print(total_data)
