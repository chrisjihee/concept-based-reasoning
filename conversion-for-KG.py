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
input_all_file = "data/YAGO3-10/edges_as_text_all.tsv"
input_test_file = "data/YAGO3-10/edges_as_text_test.tsv"
input_valid_file = "data/YAGO3-10/edges_as_text_valid.tsv"
input_train_file = "data/YAGO3-10/edges_as_text_train.tsv"
all_samples = list(tsv_lines(input_all_file))
test_samples = list(tsv_lines(input_test_file))
valid_samples = list(tsv_lines(input_valid_file))
train_samples = list(tsv_lines(input_train_file))
print(len(all_samples))
print(len(test_samples))
print(len(valid_samples))
print(len(train_samples))

grouped_data = list(grouped(all_samples, key=lambda x: x[0]))

for i, (k, v) in enumerate(grouped(all_samples, key=lambda x: x[0]), start=1):
    print(i, k, list(v))
