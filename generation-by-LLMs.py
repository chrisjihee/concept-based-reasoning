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
test_size = 3
request_timeout = 60
input_file = "data/LLM-test-with-KG-31.json"
output_file = f"data/LLM-test-with-KG-responses-{test_size}.json"
target_models = [x["full_id"] for x in load_json("conf/core_chat_models.json")]
prompt_template = read_or("template/generation_prompt.txt") or getpass("Generation Prompt: ")
api_client = Together(timeout=request_timeout,
                      api_key=read_or("conf/key-togetherai.txt") or getpass("TogetherAI API key: "))


# define function to chat with LLM
def chat_with_llm(messages, model_id):
    try:
        stream = api_client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=True,
        )
        return ''.join(chunk.choices[0].delta.content or "" for chunk in stream)
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
        difficulty = item["difficulty"]
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
        total_data.append(item)
        item["responses"] = []
        item["no_responses"] = []
        for target_model in tqdm(target_models, desc=f"* Answering question ({i}/{len(input_data)})", unit="model", file=sys.stdout):
            based = datetime.now()
            model_response = chat_with_llm(model_id=target_model,
                                           messages=[{"role": "user", "content": generation_prompt}])
            elasped = (datetime.now() - based).total_seconds()
            if model_response:
                item["responses"].append({
                    "model": target_model,
                    "output": model_response,
                    "num_words": len(model_response.split()),
                    "elasped": elasped,
                })
            else:
                item["no_responses"].append(target_model)
            save_json(total_data, output_file, indent=2, ensure_ascii=False)

# write to output file (final save)
save_json(total_data, output_file, indent=2, ensure_ascii=False)
