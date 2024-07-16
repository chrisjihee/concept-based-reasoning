import re
from getpass import getpass

from together import Together
from tqdm import tqdm

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

# setup program
test_size = 30
request_timeout = 60
default_max_tokens = 512
default_temperature = 0.5
default_repetition_penalty = 1.1
input_file = "data/LLM-test-with-KG-31.json"
output_file = f"data/LLM-test-with-KG-responses-{test_size}.json"
target_models = [x["full_id"] for x in load_json("conf/core_chat_models.json")]
prompt_template = read_or("template/generation_prompt.txt") or getpass("Generation Prompt: ")
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


# read input file
input_data = load_json(input_file)
demo, input_data = input_data[0], input_data[1:]
demo_question = demo["question"]
demo_answer = demo["answer"]
demo_answer_size = len(demo["answer"].split())
demo_knowledge_size = len(demo["triples"])
demo_triples = "\n".join([f"  - {triple}" for triple in demo["triples"]])

# chat with LLMs
with JobTimer("Answer Generation", rt=1, rb=1, rw=114, rc='=', verbose=1):
    total_data = []
    input_data = input_data[:test_size]
    for i, item in enumerate(input_data, start=1):
        total_data.append(item)
        chat_history = []
        real_question = item["question"]
        real_answer_size = len(item["answer"].split())
        real_knowledge_size = len(item["triples"])
        generation_prompt = prompt_template.format(
            real_question=real_question,
            real_answer_size=real_answer_size,
            real_knowledge_size=real_knowledge_size,
            demo_question=demo_question,
            demo_answer=demo_answer,
            demo_answer_size=demo_answer_size,
            demo_knowledge_size=demo_knowledge_size,
            demo_triples=demo_triples,
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
        for target_model in tqdm(target_models, desc=f"* Answering question ({i}/{len(input_data)})", unit="model", file=sys.stdout):
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

# write to output file (final save)
save_json(total_data, output_file, indent=2, ensure_ascii=False)
