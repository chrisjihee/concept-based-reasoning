from tqdm import tqdm

from chrisbase.data import *
from chrisbase.io import *
from chrisbase.util import *

# setup environment
logger = logging.getLogger(__name__)
args = CommonArguments(
    env=ProjectEnv(
        project="LLM-based",
        job_name="evaluation",
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
debug_test_size = 10
dataset_names = [
    "WN18RR",
    "YAGO3-10",
]
generation_levels = {
    1: "relation_only",  # Relation Classification
    # 2: "tail_only",  # Link Prediction
    # 3: "tail_with_relation",
    # 4: "free_with_quantity",
    # 5: "free_without_quantity",
}
target_generation_levels = sorted(generation_levels.keys())
successful_finish_reasons = {"stop"}

# run program
for dataset_name in dataset_names:
    for generation_level in target_generation_levels:
        extraction_file = f"extraction/{dataset_name}/edges_as_text_all-responses-{test_size}@{generation_level}.json"

        extraction_data = load_json(extraction_file)
        if debug_test_size > 0:
            extraction_data = extraction_data[:debug_test_size]
        evaluation_data = []

        with JobTimer(f"LLM Evaluation(dataset_name={dataset_name}, generation_level={generation_level}, num_extraction={len(extraction_data)})",
                      rt=1, rb=1, rw=114, rc='=', mt=1, verbose=1):
            for i, item in enumerate(tqdm(extraction_data, desc=f"* Evaluating LLM", unit="item", file=sys.stdout), start=1):
                entity = item["entity"]
                triples_by_human = normalize_triples(item["triples"])
                generation_messages = item["generation_messages"]
                extraction_messages = item["extraction_messages"]
                extraction_responses = item["responses"]

                # DEBUG PRINT
                print()
                print("=" * 100)
                print(f"entity: {entity}")
                print("-" * 100)
                print(f"triples by human: ")
                for x in triples_by_human:
                    print("  -", x)
                print("-" * 100)
                # print(f"generation_messages: \n{generation_messages[-1]['content']}")
                # print("-" * 100)
                print(f"extraction_messages: \n{extraction_messages[-1]['content']}")
                print("-" * 100)

                for j, response in enumerate(extraction_responses, start=1):
                    extraction_model = response["model"].split("/")[-1]
                    assert response["output"]["role"] == "assistant", f"role={response['output']['role']} != assistant"
                    content = str(response["output"]["content"])
                    if response["output"]["finish_reason"] in successful_finish_reasons:
                        if '[' in content and ']' in content and content.index('[') < content.rindex(']'):
                            predictions = json.loads(content[content.index('['):content.rindex(']') + 1])
                            for prediction in predictions:
                                if all([
                                    "model_id" in prediction,
                                    "triples_by_model" in prediction,
                                    isinstance(prediction["triples_by_model"], (list, tuple)),
                                ]):
                                    # print(f"prediction: {prediction}")
                                    # print("-" * 100)
                                    generation_model = prediction["model_id"]
                                    triples_by_model = normalize_triples(prediction["triples_by_model"])
                                    print(f"triples by {generation_model}:")
                                    for x in triples_by_model:
                                        print("  -", x)
                                    prec, rec, f1 = measure_performance(triples_by_human, triples_by_model)
                                    print(f"  => Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
                                    print("-" * 100)