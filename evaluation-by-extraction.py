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
        job_name="evaluation-by-extraction",
        msg_level=logging.INFO,
        msg_format=LoggingFormat.BRIEF_00,
    )
)


# define functions
def normalize_element(e):
    return str(e).replace(' ', '_').lower()

def normalize_triples(triples):
    return [[normalize_element(e) for e in triple] for triple in triples if len(triple) == 3]

def measure_performance(triples_by_human, triples_by_model):
    # Define the true labels and predicted labels
    y_test = [f'{h} -> {r} -> {t}' for h, r, t in triples_by_human]
    y_pred = [f'{h} -> {r} -> {t}' for h, r, t in triples_by_model]

    # Convert to sets for handling as a multilabel problem (ignoring order)
    y_test_set = set(y_test)
    y_pred_set = set(y_pred)

    # Precision, Recall, F1 score calculation
    prec = len(y_test_set.intersection(y_pred_set)) / len(y_pred_set) if len(y_pred_set) > 0 else 0.0
    rec = len(y_test_set.intersection(y_pred_set)) / len(y_test_set) if len(y_test_set) > 0 else 1.0
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


# setup program
test_size = 100
debug_test_size = -1
dataset_names = [
    "WN18RR",
    "YAGO3-10",
]
generation_levels = {
    1: "relation_only",  # Relation Classification
    2: "tail_only",  # Link Prediction
    3: "tail_with_relation",
    4: "free_with_quantity",
    5: "free_without_quantity",
}
target_generation_levels = sorted(generation_levels.keys())
successful_narrator_roles = {"ASSISTANT"}
successful_finish_reasons = {"stop", "eos"}
JSON_FORMAT_ERROR = "JSON format error"
JSON_RANGE_ERROR = "JSON range error"
JSON_KEY_ERROR = "JSON key error"
BASIC_EXCEPTIONS = [JSON_RANGE_ERROR, JSON_FORMAT_ERROR, JSON_KEY_ERROR]

# run program
for dataset_name in dataset_names:
    for generation_level in target_generation_levels:
        extraction_file = f"extraction/{dataset_name}/edges_as_text_all-responses-{test_size}@{generation_level}.json"
        evaluation_file = f"evaluation/{dataset_name}/{args.env.job_name}-{test_size}@{generation_level}.xlsx"
        extraction_data = load_json(extraction_file)
        if debug_test_size > 0:
            extraction_data = extraction_data[:debug_test_size]

        with JobTimer(f"LLM Evaluation(dataset_name={dataset_name}, generation_level={generation_level}, num_extraction={len(extraction_data)})",
                      rt=1, rb=1, rw=114, rc='=', mt=1, verbose=1):

            evaluation_data = []
            for i, sample in enumerate(tqdm(extraction_data, desc=f"* Evaluating LLM", unit="sample", file=sys.stdout), start=1):
                entity = sample["entity"]
                triples_by_human = normalize_triples(sample["triples"])
                extraction_messages = sample["extraction_messages"]
                extraction_outputs = sample["responses"]

                # DEBUG PRINT
                # print()
                # print("=" * 100)
                # print(f"entity: {entity}")
                # print("-" * 100)
                # print(f"triples by human: ")
                # for x in triples_by_human:
                #     print("  -", x)
                # print("-" * 100)
                # print(f"extraction_messages: \n{extraction_messages[-1]['content']}")
                # print("-" * 100)

                for j, extraction_output in enumerate(extraction_outputs, start=1):
                    extraction_model = extraction_output["model"].split("/")[-1]
                    content = str(extraction_output["output"]["content"])
                    # print(f"content: {content}")
                    finish_reason = extraction_output["output"]["finish_reason"]
                    narrator_role = extraction_output['output']['role'].upper()
                    if narrator_role in successful_narrator_roles:
                        if finish_reason in successful_finish_reasons:
                            if '[' in content and ']' in content and content.index('[') < content.rindex(']'):
                                try:
                                    predictions = json.loads(content[content.index('['):content.rindex(']') + 1])
                                    # print(f"predictions: {predictions}")
                                    for prediction in predictions:
                                        generation_model = prediction["model_id"] if "model_id" in prediction else "unknown"
                                        if "triples_by_model" in prediction and isinstance(prediction["triples_by_model"], (list, tuple)):
                                            triples_by_model = normalize_triples(prediction["triples_by_model"])
                                            prec, rec, f1 = measure_performance(triples_by_human, triples_by_model)
                                            evaluation_data.append({
                                                "i": i,
                                                "j": j,
                                                "type": "generation",
                                                "model": generation_model,
                                                "prec": prec,
                                                "rec": rec,
                                                "f1": f1,
                                                "exception": np.nan,
                                            })
                                        else:
                                            evaluation_data.append({
                                                "i": i,
                                                "j": j,
                                                "type": "generation",
                                                "model": generation_model,
                                                "exception": JSON_KEY_ERROR,
                                            })
                                except JSONDecodeError:
                                    evaluation_data.append({
                                        "i": i,
                                        "j": j,
                                        "type": "extraction",
                                        "model": extraction_model,
                                        "exception": JSON_FORMAT_ERROR,
                                    })
                            else:
                                evaluation_data.append({
                                    "i": i,
                                    "j": j,
                                    "type": "extraction",
                                    "model": extraction_model,
                                    "exception": JSON_RANGE_ERROR
                                })
                        else:
                            evaluation_data.append({
                                "i": i,
                                "j": j,
                                "type": "extraction",
                                "model": extraction_model,
                                "exception": f"{narrator_role}: {finish_reason}"
                            })
                    else:
                        evaluation_data.append({
                            "i": i,
                            "j": j,
                            "type": "extraction",
                            "model": extraction_model,
                            "exception": f"{narrator_role}: {finish_reason}"
                        })

            evaluation_data = pd.DataFrame(evaluation_data)
            evaluation_summary = evaluation_data.groupby(['type', 'model']).agg(
                prec_mean=('prec', 'mean'),
                rec_mean=('rec', 'mean'),
                f1_mean=('f1', 'mean'),
                valid_count=('f1', lambda x: x.notna().sum()),
                invalid_count=('exception', 'count'),
                exception_counts=('exception', lambda x: x.value_counts().to_dict())
            ).reset_index().sort_values(by=['type', 'model'])

            exception_counts = evaluation_summary['exception_counts'].apply(pd.Series).fillna(0).astype(int)
            exception_counts = exception_counts.reindex(
                columns=BASIC_EXCEPTIONS + sorted([col for col in exception_counts.columns if col not in BASIC_EXCEPTIONS]),
                fill_value=0
            )
            evaluation_summary = pd.concat([evaluation_summary.drop(columns=['exception_counts']), exception_counts], axis=1)

            logger.info(f"evaluation_summary: \n{evaluation_summary}")
            evaluation_summary.to_excel(make_parent_dir(evaluation_file), index=False)
