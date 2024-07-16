from tqdm import tqdm
from getpass import getpass

from together import Together

from chrisbase.io import *
from chrisbase.util import *

TOGETHER_API_TOKEN = read_or(first_path_or("together-tokens*")) or getpass()
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_TOKEN

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))


def chat_with_llm(messages, model="meta-llama/Llama-3-8b-chat-hf"):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    return ''.join(chunk.choices[0].delta.content or "" for chunk in stream)


# setup target models
target_models = [
    "google/gemma-2b-it",
    "google/gemma-7b-it",
    "togethercomputer/alpaca-7b",
    "lmsys/vicuna-7b-v1.5",
    "lmsys/vicuna-13b-v1.5",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-3-8b-chat-hf",
    "meta-llama/Llama-3-70b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "upstage/SOLAR-10.7B-Instruct-v1.0",
    "Qwen/Qwen1.5-0.5B-Chat",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen1.5-14B-Chat",
    "Qwen/Qwen1.5-32B-Chat",
    "Qwen/Qwen1.5-72B-Chat",
    "Qwen/Qwen1.5-110B-Chat",
    "Qwen/Qwen2-72B-Instruct",
]

# read a json file
input_file = "data/LLM-test-with-KG-31.json"
output_file = "data/LLM-test-with-KG-responses.json"
test_set = load_json(input_file)
demo_set, test_set = test_set[:1], test_set[1:]

prompt_template = """
The following is a test to compare how well large language models (LLMs) structure and store knowledge through knowledge graphs (KGs).
A knowledge graph typically represents knowledge in the form of a triple (subject -> relation -> object), and the following is an example of such a representation.

-----
Marie Curie -> Born -> November 7, 1867, Warsaw, Poland
Marie Curie -> Died -> July 4, 1934 (age 66 years), Passy, France
Marie Curie -> Discovered -> Radium, Polonium
Marie Curie -> Spouse -> Pierre Curie (m. 1895–1906)
Marie Curie -> Buried -> April 20, 1995, Panthéon, Paris, France
Marie Curie -> Education -> University of Paris (1903), University of Paris (1894), MORE
Marie Curie -> Children -> Ève Curie, Irène Joliot-Curie
-----

We want to check your knowledge structuring ability to ensure that you include this knowledge graph well.
Print an answer of around {real_answer_size} words to a given question, along with a triple set containing about {real_knowledge_size} different relationships that reference the answer.
* Question: {real_question}
* Answer(about {real_answer_size} words): (your answer here)
* Related triples(about {real_knowledge_size} relationships): (your knowledge graph triples here) 

Below is a demo example of an answer to a question that can be answered based on the knowledge graph above, and a set of related triples to reference in the answer.
* Question: {demo_question}
* Answer(about {demo_answer_size} words): {demo_answer}
* Related triples(about {demo_knowledge_size} relationships):
{demo_triples}
"""

demo = demo_set[0]
demo_question = demo["question"]
demo_answer = demo["answer"]
demo_answer_size = len(demo["answer"].split())
demo_knowledge_size = len(demo["triples"])
demo_triples = "\n".join([f"  - {triple}" for triple in demo["triples"]])

total_responses = []
for i, qa in enumerate(test_set[:2], start=1):
    difficulty = qa["difficulty"]
    real_question = qa["question"]
    real_answer_size = len(qa["answer"].split())
    real_knowledge_size = len(qa["triples"])
    prompt = prompt_template.format(
        real_question=real_question,
        real_answer_size=real_answer_size,
        real_knowledge_size=real_knowledge_size,
        demo_question=demo_question,
        demo_answer=demo_answer,
        demo_answer_size=demo_answer_size,
        demo_knowledge_size=demo_knowledge_size,
        demo_triples=demo_triples,
    )
    model_responses = []
    for model in tqdm(target_models, desc=f"* Answering question ({i}/{len(test_set)})", unit="model"):
        r = {
            "model": model,
            "output": chat_with_llm(model=model,
                                    messages=[{"role": "user", "content": prompt}]),
        }
        model_responses.append(r)
    qa["responses"] = model_responses
    total_responses.append(qa)
save_json(total_responses, output_file)
