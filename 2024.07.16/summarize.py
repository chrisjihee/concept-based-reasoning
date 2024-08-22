import matplotlib.pyplot as plt
import numpy as np

from chrisbase.io import *

"""
example record:
{
    "number": 1,
    "model_id": "Llama-2-7b-chat-hf",
    "answer_score": { "sum": 52.0, "accuracy": 7.0, "completeness": 6.5, "relevance": 7.0, "consistency": 7.0, "detail": 6.5, "sophistication": 6.5, "hierarchy": 5.5, "context": 6.0 },
    "triples_score": { "sum": 53.5, "accuracy": 7.5, "completeness": 6.0, "relevance": 7.0, "consistency": 7.5, "detail": 6.5, "sophistication": 6.0, "hierarchy": 6.5, "context": 6.5 }
}
"""

model_id_map = {
    "Llama-2-07B-chat-hf": "Llama-2-7b-chat-hf",
    "Llama-2-13B-chat-hf": "Llama-2-13b-chat-hf",
    "Llama-2-70B-chat-hf": "Llama-2-70b-chat-hf",
    "Llama-3-08B-chat-hf": "Llama-3-8b-chat-hf",
    "Llama-3-70B-chat-hf": "Llama-3-70b-chat-hf",
    "Mistral-7B-Instruct-v0.1": "Mistral-7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct-v0.2",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B-Instruct-v0.3",
    "Mistral-7B-OpenOrca": "Mistral-7B-OpenOrca",
    "Mixtral-8x07B-Instruct-v0.1": "Mixtral-8x7B-Instruct-v0.1",
    "Mixtral-8x22B-Instruct-v0.1": "Mixtral-8x22B-Instruct-v0.1",
    "OpenChat-3.5-1210": "openchat-3.5-1210",
    "Qwen1.5-000.5B-Chat": "Qwen1.5-0.5B-Chat",
    "Qwen1.5-001.8B-Chat": "Qwen1.5-1.8B-Chat",
    "Qwen1.5-004.0B-Chat": "Qwen1.5-4B-Chat",
    "Qwen1.5-007.0B-Chat": "Qwen1.5-7B-Chat",
    "Qwen1.5-014.0B-Chat": "Qwen1.5-14B-Chat",
    "Qwen1.5-032.0B-Chat": "Qwen1.5-32B-Chat",
    "Qwen1.5-072.0B-Chat": "Qwen1.5-72B-Chat",
    "Qwen1.5-110.0B-Chat": "Qwen1.5-110B-Chat",
    "Qwen2.0-072B-Instruct": "Qwen2-72B-Instruct",
    "SOLAR-10.7B-Instruct-v1.0": "SOLAR-10.7B-Instruct-v1.0",
    "Vicuna-07B-v1.5": "vicuna-7b-v1.5",
    "Vicuna-13B-v1.5": "vicuna-13b-v1.5",
    "WizardLM-13B-V1.2": "WizardLM-13B-V1.2",
}
model_id_map = {v: k for k, v in model_id_map.items()}

# Load the evaluation results
aspects = ["sum", "accuracy", "completeness", "relevance", "consistency", "detail", "sophistication", "hierarchy", "context"]
for aspect in aspects:
    scores_per_model = {}
    scores_for_total = {}
    for file in files("eval/LLM-test-with-KG-evaluation-result-*.json"):
        n = int(file.stem.split("-")[-1])
        for item in load_json(file):
            item["model_id"] = model_id_map[item["model_id"]]
            if item["model_id"] not in scores_per_model:
                scores_per_model[item["model_id"]] = {"answer_per": {}, "triples_per": {}}
            scores_per_model[item["model_id"]]["answer_per"][n] = item["answer_score"][aspect]
            scores_per_model[item["model_id"]]["triples_per"][n] = item["triples_score"][aspect]
            scores_per_model[item["model_id"]]["answer_avg"] = sum(scores_per_model[item["model_id"]]["answer_per"].values()) / len(scores_per_model[item["model_id"]]["answer_per"])
            scores_per_model[item["model_id"]]["triples_avg"] = sum(scores_per_model[item["model_id"]]["triples_per"].values()) / len(scores_per_model[item["model_id"]]["triples_per"])
            model_qa_id = f"{item['model_id']}={n:02d}"
            if model_qa_id not in scores_for_total:
                scores_for_total[model_qa_id] = {}
            scores_for_total[model_qa_id]["answer"] = item["answer_score"][aspect]
            scores_for_total[model_qa_id]["triples"] = item["triples_score"][aspect]

    # Create a DataFrame to store the results for average
    results = []
    for model_id, scores in scores_per_model.items():
        results.append({
            "Model ID": model_id,
            f"Avg Answer({aspect})": scores["answer_avg"],
            f"Avg Triples({aspect})": scores["triples_avg"]
        })
    results_df = pd.DataFrame(results)
    results_df.sort_values(inplace=True, by="Model ID")
    results_df.reset_index(inplace=True, drop=True)
    results_df.to_excel(f"out/scores_per_model({aspect}).xlsx", index=False)
    print(results_df)

    # Plot the scatter plot for average scores
    plt.figure(figsize=(12, 8))
    colormap = plt.get_cmap('turbo', len(results_df['Model ID'].unique()))
    colors = {model_id: colormap(i) for i, model_id in enumerate(results_df['Model ID'].unique())}
    for model_id in results_df['Model ID'].unique():
        subset = results_df[results_df['Model ID'] == model_id]
        jitter_x = np.random.normal(0, 0.3, size=subset.shape[0])
        jitter_y = np.random.normal(0, 0.3, size=subset.shape[0])
        plt.scatter(subset[f"Avg Answer({aspect})"] + jitter_x, subset[f"Avg Triples({aspect})"] + jitter_y, color=colors[model_id], label=model_id)
    plt.title(f'Correlation between Avg Answer({aspect}) and Avg Triples({aspect})')
    plt.xlabel(f"Avg Answer({aspect})")
    plt.ylabel(f"Avg Triples({aspect})")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', fontsize='small', title_fontsize='small', borderaxespad=0., ncol=1)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f"out/correlation-per-model({aspect}).png")

    # Create a DataFrame to store the results for total
    results = []
    for model_qa_id, scores in scores_for_total.items():
        model_id = model_qa_id.split("=")[0]  # Extract model_id from model_qa_id
        results.append({
            "Model QA ID": model_qa_id,
            "Model ID": model_id,
            f"All Answer({aspect})": scores["answer"],
            f"All Triples({aspect})": scores["triples"]
        })
    results_df = pd.DataFrame(results)
    results_df.sort_values(inplace=True, by="Model QA ID")
    results_df.reset_index(inplace=True, drop=True)
    results_df.to_excel(f"out/scores_all_model({aspect}).xlsx", index=False)

    # Plot the scatter plot for average scores
    plt.figure(figsize=(12, 8))
    colormap = plt.get_cmap('turbo', len(results_df['Model ID'].unique()))
    colors = {model_id: colormap(i) for i, model_id in enumerate(results_df['Model ID'].unique())}
    for model_id in results_df['Model ID'].unique():
        subset = results_df[results_df['Model ID'] == model_id]
        jitter_x = np.random.normal(0, 0.1, size=subset.shape[0])
        jitter_y = np.random.normal(0, 0.1, size=subset.shape[0])
        plt.scatter(subset[f"All Answer({aspect})"] + jitter_x, subset[f"All Triples({aspect})"] + jitter_y, color=colors[model_id], label=model_id)
    plt.title(f'Correlation between All Answer({aspect}) and All Triples({aspect})')
    plt.xlabel(f"All Answer({aspect})")
    plt.ylabel(f"All Triples({aspect})")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', fontsize='small', title_fontsize='small', borderaxespad=0., ncol=1)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f"out/correlation-all-model({aspect}).png")
