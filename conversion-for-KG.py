import re
from getpass import getpass

from sklearn.model_selection import train_test_split

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
dataset = "WN18RR"
input_all_file = f"data/{dataset}/edges_as_text_all.tsv"
test_size = 100
debug_test_size = 1
demo_size_per_size = 1
prompt_template = read_or("template/generation_prompt.txt") or getpass("Generation Prompt: ")
demo_prompt_match = re.search(r'<BEGIN_OF_DEMO_EXAMPLE>(?s:.)+<END_OF_DEMO_EXAMPLE>', prompt_template)
demo_prompt_start = demo_prompt_match.start()
demo_prompt_end = demo_prompt_match.end()
demo_template = prompt_template[demo_prompt_start: demo_prompt_end]

all_samples = list(tsv_lines(input_all_file))
total_data = [{"entity": k, "triples": list(v)} for k, v in grouped(all_samples, key=lambda x: x[0])]
train_data, test_data = train_test_split(total_data, test_size=test_size, random_state=7)
print(len(train_data))
print(len(test_data))

train_data_per_size = {k: list(v) for k, v in grouped(train_data, key=lambda x: len(x["triples"]))}
test_data_per_size = {k: list(v) for k, v in grouped(test_data, key=lambda x: len(x["triples"]))}

demo_data = []
for size in test_data_per_size.keys():
    if size in train_data_per_size:
        demo_data += train_data_per_size[size][:demo_size_per_size]
print(len(demo_data))
for x in demo_data:
    print(x)

# chat with LLMs
with JobTimer("Answer Generation", rt=1, rb=1, rw=114, rc='=', verbose=1):
    total_data = []
    if debug_test_size > 0:
        test_data = test_data[:debug_test_size]
    for i, item in enumerate(test_data, start=1):
        total_data.append(item)
        chat_history = []
        test_entity = item["entity"]
        test_triples = item["triples"]
        test_triples_size = len(test_triples)

        demo_prompts = [
            demo_template.format(
                demo_entity=demo["entity"],
                demo_triples_size=len(demo["triples"]),
                demo_triples="\n".join([f"  - {s} -> {r} -> {o}" for (s, r, o) in demo["triples"]]),
            ) for demo in demo_data
        ]
        generation_prompt = prompt_template[:demo_prompt_start] + "\n\n".join(demo_prompts) + prompt_template[demo_prompt_end:].format(
            test_entity=test_entity,
            test_triples_size=test_triples_size,
        )
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

        print(generation_prompt)

print(total_data)
