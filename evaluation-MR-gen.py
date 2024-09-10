from json import JSONDecodeError

from tqdm import tqdm

from chrisbase.data import *
from chrisbase.io import *
from chrisbase.util import *

# setup environment
logger = logging.getLogger(__name__)
args = CommonArguments(
    env=ProjectEnv(
        project="LLM-based",
        job_name="evaluation-MR-gen",
        msg_level=logging.INFO,
        msg_format=LoggingFormat.BRIEF_00,
    )
)


# define functions
prefix_unit = re.compile(r"^[\"'A-Za-z_.,!?$ ]+")
suffix_unit = re.compile(r"[\"'A-Za-z_.,!?$ ]+$")


def normalize_math_answer(x: str | int | float):
    x = str(x)
    x = x.split("\r")[-1]
    x = x.split("\n")[-1]
    x = x.split(". ")[-1]
    x = x.split("! ")[-1]
    x = x.split("? ")[-1]
    x = x.split(":")[-1]
    x = x.split("=")[-1]
    x = x.strip()
    x = x.replace(",", "")
    x = prefix_unit.sub("", x) if prefix_unit.sub("", x) else x
    x = suffix_unit.sub("", x) if suffix_unit.sub("", x) else x
    x = x.strip()
    try:
        x = str(float(x))
    except ValueError:
        x = str(x)
    return x


def measure_math_reasoning(answer_by_human, answer_by_model):
    if answer_by_human == answer_by_model:
        return 1.0
    elif answer_by_human.replace(",", "") == answer_by_model.replace(",", ""):
        return 0.9
    else:
        return 0.0


# setup program
test_size = 100
debug_test_size = -1
dataset_names = [
    "GSM8k",
]
generation_levels = {
    1: "answer_only",
    2: "answer_and_explanation_with_quantity",
    # 3: "answer_and_explanation_and_equation_with_quantity",
    # 4: "answer_and_explanation_without_quantity",
    # 5: "answer_and_explanation_and_equation_without_quantity",
}
successful_narrator_roles = {"ASSISTANT"}
successful_finish_reasons = {"stop", "eos"}
JSON_FORMAT_ERROR = "JSON format error"
JSON_RANGE_ERROR = "JSON range error"
JSON_KEY_ERROR = "JSON key error"
BASIC_EXCEPTIONS = [JSON_RANGE_ERROR, JSON_FORMAT_ERROR, JSON_KEY_ERROR]

# run program
for dataset_name in dataset_names:
    for generation_level in sorted(generation_levels.keys()):
        generation_file = f"generation/{dataset_name}/GSM8k_test-by-LLM-{test_size}@{generation_level}.json"
        evaluation_file = f"evaluation/{dataset_name}/{args.env.job_name}-{test_size}@{generation_level}.xlsx"
        generation_data = load_json(generation_file)
        if debug_test_size > 0:
            generation_data = generation_data[:debug_test_size]

        with JobTimer(f"LLM Evaluation(dataset_name={dataset_name}, generation_level={generation_level}, num_generation={len(generation_data)})",
                      rt=1, rb=1, rw=114, rc='=', mt=1, verbose=1):

            evaluation_data = []
            for i, sample in enumerate(tqdm(generation_data, desc=f"* Evaluating LLM", unit="sample", file=sys.stdout), start=1):
                problem = sample["problem"]
                answer_by_human = normalize_math_answer(sample["final_answer"])
                reasoning_by_human = sample["reasoning_by_human"]
                generation_messages = sample["generation_messages"]
                generation_outputs = sample["generation_outputs"]

                # DEBUG PRINT
                # print()
                # print("=" * 100)
                # print(f"problem: {problem}")
                # print(f'answer_by_human: {sample["final_answer"]} -> {answer_by_human}')
                # print("-" * 100)
                # print(f"reasoning_by_human: ")
                # for x in reasoning_by_human:
                #     print("  -", x)
                # print("-" * 100)
                # print(f"generation_messages: \n{generation_messages[-1]['content']}")
                # print("-" * 100)

                for j, generation_output in enumerate(generation_outputs, start=1):
                    generation_type = generation_output["type"]
                    generation_model = generation_output["model"].split("/")[-1]
                    output_content = str(generation_output["output"]["content"])
                    finish_reason = generation_output["output"]["finish_reason"]
                    narrator_role = generation_output['output']['role'].upper()
                    # print(f"output_content: {output_content}")
                    # print(f"finish_reason: {finish_reason}")
                    # print(f"narrator_role: {narrator_role}")
                    if narrator_role in successful_narrator_roles:
                        if finish_reason in successful_finish_reasons:
                            if '{' in output_content and '}' in output_content and output_content.index('{') < output_content.rindex('}'):
                                try:
                                    prediction = json.loads(output_content[output_content.index('{'):output_content.rindex('}') + 1])
                                    # print(f"prediction: {json.dumps(prediction, indent=2)}")
                                    if "final_answer" in prediction and isinstance(prediction["final_answer"], (str, int, float)):
                                        answer_by_model = normalize_math_answer(prediction["final_answer"])
                                        # print(f'answer_by_model({generation_model:40s}): {prediction["final_answer"]} -> {answer_by_model}')
                                        acc = measure_math_reasoning(answer_by_human, answer_by_model)
                                        evaluation_data.append({
                                            "i": i,
                                            "j": j,
                                            "type": generation_type,
                                            "model": generation_model,
                                            "acc": acc,
                                            "exception": np.nan,
                                        })
                                    else:
                                        evaluation_data.append({
                                            "i": i,
                                            "j": j,
                                            "type": generation_type,
                                            "model": generation_model,
                                            "exception": JSON_KEY_ERROR,
                                        })
                                except JSONDecodeError:
                                    evaluation_data.append({
                                        "i": i,
                                        "j": j,
                                        "type": generation_type,
                                        "model": generation_model,
                                        "exception": JSON_FORMAT_ERROR,
                                    })
                            else:
                                evaluation_data.append({
                                    "i": i,
                                    "j": j,
                                    "type": generation_type,
                                    "model": generation_model,
                                    "exception": JSON_RANGE_ERROR
                                })
                        else:
                            evaluation_data.append({
                                "i": i,
                                "j": j,
                                "type": generation_type,
                                "model": generation_model,
                                "exception": f"{narrator_role}: {finish_reason}"
                            })
                    else:
                        evaluation_data.append({
                            "i": i,
                            "j": j,
                            "type": generation_type,
                            "model": generation_model,
                            "exception": f"{narrator_role}: {finish_reason}"
                        })

            evaluation_data = pd.DataFrame(evaluation_data)
            evaluation_summary = evaluation_data.groupby(['model', 'type']).agg(
                acc_mean=('acc', 'mean'),
                valid_count=('acc', lambda x: x.notna().sum()),
                invalid_count=('exception', 'count'),
                exception_counts=('exception', lambda x: x.value_counts().to_dict())
            ).reset_index().sort_values(by=['model', 'type'])

            exception_counts = evaluation_summary['exception_counts'].apply(pd.Series).fillna(0).astype(int)
            exception_counts = exception_counts.reindex(
                columns=BASIC_EXCEPTIONS + sorted([col for col in exception_counts.columns if col not in BASIC_EXCEPTIONS]),
                fill_value=0
            )
            evaluation_summary = pd.concat([evaluation_summary.drop(columns=['exception_counts']), exception_counts], axis=1)

            logger.info(f"evaluation_summary: \n{evaluation_summary}")
            evaluation_summary.to_excel(make_parent_dir(evaluation_file), index=False)
