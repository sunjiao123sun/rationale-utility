from model.simplet5 import SimpleT5
from utils import *
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", "-i", help="the path for training data",
                    default="../processed-data/cose-v1-11/train.csv")
parser.add_argument("--test_path", "-t", help="the path for the test data",
                    default="../processed-data/cose-v1-11/dev.csv")
parser.add_argument("--data_record", "-d", default="../hf-prepare-i2r")
parser.add_argument("--output_dir", "-o", help="the outputdir for the model")
parser.add_argument("--gpus", "-g", default="[0]")
parser.add_argument("--batch_size", "-bs", default=8, type=int)
parser.add_argument("--max_epochs", "-epoch", default=30, type=int)
parser.add_argument("--model_name", "-m", choices=["t5-base", "allenai/unifiedqa-t5-base"])
# parser.add_argument("--rationale", "-r", help="if add rationale in the input", default=False)


args = parser.parse_args()

dev_path = args.test_path
train_path = args.train_path
data_record = args.data_record
output_dir = args.output_dir
gpus = ast.literal_eval(args.gpus)
model_name = args.model_name
# use_rationale = args.rationale

if "combine" not in train_path:
    train_df = i2r_variant(train_path, data_record)
    dev_df = i2r_variant(dev_path, data_record)
else:
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)


model = SimpleT5()
model.from_pretrained(model_type="t5", model_name=model_name)
model.train(train_df=train_df,
            eval_df=dev_df,
            source_max_token_len=128,
            target_max_token_len=50,
            batch_size=8,
            max_epochs=30,
            use_gpu=True,
            gpus=gpus,
            outputdir=output_dir)
