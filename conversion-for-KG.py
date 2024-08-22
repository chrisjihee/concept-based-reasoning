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
demo_size_per_size = 1

all_samples = list(tsv_lines(input_all_file))
grouped_heads = [list(v) for k, v in grouped(all_samples, key=lambda x: x[0])]
train_data, test_data = train_test_split(grouped_heads, test_size=test_size, random_state=7)
print(len(train_data))
print(len(test_data))

train_data_per_size = {k: list(v) for k, v in grouped(train_data, key=lambda x: len(x))}
test_data_per_size = {k: list(v) for k, v in grouped(test_data, key=lambda x: len(x))}

demo_data = []
for size in test_data_per_size.keys():
    if size in train_data_per_size:
        demo_data += train_data_per_size[size][:demo_size_per_size]
print(len(demo_data))
