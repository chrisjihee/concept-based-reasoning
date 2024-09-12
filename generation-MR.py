from getpass import getpass
from typing import ClassVar

from openai import OpenAI
from pydantic import BaseModel
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
        job_name="generation-MR",
        msg_level=logging.INFO,
        msg_format=LoggingFormat.BRIEF_00,
    )
)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = read_or("conf/key-openai-default.txt") or getpass("OpenAI API key: ")
openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
if "TOGETHER_API_KEY" not in os.environ:
    os.environ["TOGETHER_API_KEY"] = read_or("conf/key-togetherai.txt") or getpass("TogetherAI API key: ")
together_client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))


# define function to parse with LLM through OpenAI
class ReasoningStep(BaseModel):
    equation_pattern: ClassVar[re.Pattern] = re.compile(r"<<(.+?)>>")
    explanation: str
    equation: Optional[str] = None

    @classmethod
    def to_reasoning_steps(cls, solution: str) -> list["ReasoningStep"]:
        rs = []
        for x in solution.splitlines():
            e = cls.equation_pattern.findall(x)
            e = e[0] if e else ""
            x = cls.equation_pattern.sub("", x)
            if x.startswith("#### "):
                continue
            rs.append(ReasoningStep(explanation=x, equation=e))
        return rs


class MathWordProblem(BaseModel):
    problem: str
    reasoning_steps: Optional[list[ReasoningStep]]
    final_answer: str


def parse_with_LLM_by_OpenAI(response_format, **kwargs):
    try:
        response = openai_client.beta.chat.completions.parse(
            response_format=response_format,
            **kwargs,
        )
        choice = response.choices[0]
        return {
            "role": choice.message.role,
            "content": choice.message.parsed.strip(),
            "finish_reason": choice.finish_reason,
        }
    except Exception as e:
        return {
            "role": "report",
            "content": str(e),
            "finish_reason": f"{type(e).__name__}",
        }


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
num_demo_sample = 3
dataset_names = [
    "GSM8k",
]
generation_levels = {
    1: "answer_only",
    2: "answer_and_explanation_with_quantity",
    3: "answer_and_explanation_and_equation_with_quantity",
    4: "answer_and_explanation_without_quantity",
    5: "answer_and_explanation_and_equation_without_quantity",
}
generation_models = [
    ("mistralai/Mistral-7B-Instruct-v0.1", "text", None),
    ("mistralai/Mistral-7B-Instruct-v0.2", "text", None),
    ("mistralai/Mistral-7B-Instruct-v0.3", "text", None),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", "text", None),
    ("mistralai/Mixtral-8x22B-Instruct-v0.1", "text", None),
    ("meta-llama/Meta-Llama-3-8B-Instruct-Turbo", "text", None),
    ("meta-llama/Meta-Llama-3-70B-Instruct-Turbo", "text", None),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "text", None),
    ("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "text", None),
    ("meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", "text", None),
    # ("gpt-4o-mini-2024-07-18", "text", None),
    # ("gpt-4o-mini-2024-07-18", "json", MathWordProblem),
    # ("gpt-4o-2024-08-06", "text", None),
    # ("gpt-4o-2024-08-06", "json", MathWordProblem),
]
max_tokens = 4000
system_prompt = "You are a helpful math tutor. Guide the user through the solution step by step. Respond in JSON format."
generation_prompt = read_or("template/generation-MR.txt") or getpass("Generation MR Prompt: ")
random_seed = 70

# run program
for dataset_name in dataset_names:
    dataset_test_file = Path(f"data/{dataset_name}/GSM8k_test.json")
    dataset_train_file = Path(f"data/{dataset_name}/GSM8k_train.json")
    test_data = shuffled(load_json(dataset_test_file).values(), seed=random_seed)[:test_size]
    train_data = list(load_json(dataset_train_file).values())
    if debug_test_size > 0:
        test_data = test_data[:debug_test_size]
    # print(f"train_data: {len(train_data)}")
    # print(f"test_data: {len(test_data)}")

    for generation_level in sorted(generation_levels.keys()):
        generation_file = f"generation/{dataset_name}/{dataset_test_file.stem}-by-LLM-{test_size}@{generation_level}.json"
        generation_data = []

        demo_examples = []
        for sample in shuffled(train_data, seed=random_seed)[:num_demo_sample]:
            problem = sample["problem"]
            final_answer = sample["answer"]
            solution = sample["solution"]
            reasoning_by_human: list[ReasoningStep] = ReasoningStep.to_reasoning_steps(solution)
            reasoning_for_demo = []
            if generation_level == 1:
                reasoning_for_demo = None
            elif generation_level == 2:
                for step in reasoning_by_human:
                    reasoning_for_demo.append(ReasoningStep(explanation=step.explanation))
            elif generation_level == 3:
                for step in reasoning_by_human:
                    reasoning_for_demo.append(ReasoningStep(explanation=step.explanation, equation=step.equation))
            elif generation_level == 4:
                step = reasoning_by_human[0]
                reasoning_for_demo.append(ReasoningStep(explanation=step.explanation))
                reasoning_for_demo.append(ReasoningStep(explanation="..."))
            elif generation_level == 5:
                step = reasoning_by_human[0]
                reasoning_for_demo.append(ReasoningStep(explanation=step.explanation, equation=step.equation))
                reasoning_for_demo.append(ReasoningStep(explanation="...", equation="..."))
            else:
                assert False, f"Invalid generation_level: {generation_level}"
            actual_generation_demo = MathWordProblem(
                problem=problem,
                reasoning_steps=reasoning_for_demo,
                final_answer=final_answer,
            ).model_dump_json(indent=2, exclude_none=True)
            demo_examples.append(actual_generation_demo)

        with JobTimer(f"MR Generation(dataset_name={dataset_name}, generation_level={generation_level}, num_test={len(test_data)}, generation_models={len(generation_models)}, max_tokens={max_tokens})",
                      rt=1, rb=1, rw=114, rc='=', mt=1, verbose=1):
            for i, sample in enumerate(test_data, start=1):
                problem = sample["problem"]
                final_answer = sample["answer"]
                solution = sample["solution"]
                reasoning_by_human: list[ReasoningStep] = ReasoningStep.to_reasoning_steps(solution)
                reasoning_by_model = []
                if generation_level == 1:
                    reasoning_by_model = None
                elif generation_level == 2:
                    for step in reasoning_by_human:
                        reasoning_by_model.append(ReasoningStep(explanation="(explanation)"))
                elif generation_level == 3:
                    for step in reasoning_by_human:
                        reasoning_by_model.append(ReasoningStep(explanation="(explanation)", equation="(equation)"))
                elif generation_level == 4:
                    reasoning_by_model.append(ReasoningStep(explanation="(explanation)"))
                    reasoning_by_model.append(ReasoningStep(explanation="..."))
                elif generation_level == 5:
                    reasoning_by_model.append(ReasoningStep(explanation="(explanation)", equation="(equation)"))
                    reasoning_by_model.append(ReasoningStep(explanation="...", equation="..."))
                else:
                    assert False, f"Invalid generation_level: {generation_level}"
                actual_generation_prompt = generation_prompt.format(
                    generation_demo_examples="\n\n".join(f"<demo>\n{x}\n</demo>" for x in demo_examples),
                    generation_form=MathWordProblem(
                        problem=problem,
                        reasoning_steps=reasoning_by_model,
                        final_answer="(final_answer)",
                    ).model_dump_json(indent=2, exclude_none=True),
                )
                generation_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": actual_generation_prompt},
                ]
                generation_result = {
                    "dataset_name": dataset_name,
                    "problem": problem,
                    "final_answer": final_answer,
                    "reasoning_by_human": [x.model_dump() for x in reasoning_by_human],
                    "generation_level": generation_level,
                    "generation_messages": generation_messages,
                    "generation_outputs": [],
                    "generation_errors": [],
                }
                generation_data.append(generation_result)
                # print("\n" * 3)
                # print(f'<problem>\n{problem}\n</problem>')
                # print(f'<final_answer>\n{final_answer}\n</final_answer>')
                # print(f'<reasoning_by_human>\n{json.dumps([x.model_dump() for x in reasoning_by_human], indent=2, ensure_ascii=False)}\n</reasoning_by_human>')
                # print("\n" * 3)
                # print(f'<generation_level>\n{generation_level}\n</generation_level>')
                # print(f'<generation_messages>\n{generation_messages}\n</generation_messages>')
                # print(f'<actual_generation_prompt>\n{actual_generation_prompt}\n</actual_generation_prompt>')
                # print("\n" * 3)
                # exit(1)
                for (generation_model, generation_type, generation_schema) in tqdm(generation_models, desc=f"* Generating MR ({i}/{len(test_data)})", unit="model", file=sys.stdout):
                    based = datetime.now()
                    if generation_model.startswith("gpt-"):
                        if generation_type == "json":
                            if generation_schema:
                                generation_output = parse_with_LLM_by_OpenAI(
                                    model=generation_model,
                                    messages=generation_messages,
                                    max_tokens=max_tokens,
                                    response_format=generation_schema,
                                )
                            else:
                                generation_output = chat_with_LLM_by_OpenAI(
                                    model=generation_model,
                                    messages=generation_messages,
                                    max_tokens=max_tokens,
                                    response_format={"type": "json_object"},
                                )
                        else:
                            generation_output = chat_with_LLM_by_OpenAI(
                                model=generation_model,
                                messages=generation_messages,
                                max_tokens=max_tokens,
                            )
                    else:
                        if generation_type == "json":
                            generation_output = chat_with_LLM_by_Together(
                                model=generation_model,
                                messages=generation_messages,
                                max_tokens=max_tokens,
                                response_format={"type": "json_object"},
                            )
                        else:
                            generation_output = chat_with_LLM_by_Together(
                                model=generation_model,
                                messages=generation_messages,
                                max_tokens=max_tokens,
                            )
                    generation_seconds = (datetime.now() - based).total_seconds()
                    if generation_output and generation_output["content"]:
                        content_len = len(str(generation_output["content"]))
                        generation_result["generation_outputs"].append({
                            "type": generation_type,
                            "model": generation_model,
                            "output": generation_output,
                            "seconds": generation_seconds,
                            "content_len": content_len,
                        })
                    else:
                        generation_result["generation_errors"].append({
                            "type": generation_type,
                            "model": generation_model,
                            "output": generation_output,
                            "seconds": generation_seconds,
                        })
                    # print("\n" * 3)
                    # print("=" * 200)
                    # print(f'<generation_type>{generation_type}</generation_type>')
                    # print(f'<generation_model>{generation_model}</generation_model>')
                    # if generation_output and "content" in generation_output:
                    #     print(f'<generation_output_content>\n{generation_output["content"]}\n</generation_output_content>')
                    # print("=" * 200)
                    save_json(generation_data, generation_file, indent=2, ensure_ascii=False)

        save_json(generation_data, generation_file, indent=2, ensure_ascii=False)
