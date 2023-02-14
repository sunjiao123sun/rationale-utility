from model.simplet5 import SimpleT5
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", "-m", help="the path for the trained model",
                    default="/nas/home/jiaosun/rationale-eval/i2r-result/i2r-i2r-ecqa-len-512-ir2o-simplet5-epoch-29-train-loss-0.0037.csv")
parser.add_argument("--test_path", "-t", help="the path for the test data",
                    default="../hf-prepare-i2o/cose-v1-0-True/dev.csv")
parser.add_argument("--result", "-dr", default="../i2o-hf-test-result/")

args = parser.parse_args()

model_path = args.model_path
output_file = open(f"{args.result}/{model_path.split('/')[-2]}-{model_path.split('/')[-1]}.txt",
                   "w+")


model = SimpleT5()
model.load_model("t5", model_path, use_gpu=True)

df = pd.read_csv(args.test_path)
HIT = 0
for i, row in df.iterrows():
    source_text = row['source_text']
    target_text = row['target_text'].strip()
    prediction = model.predict(source_text)[0].strip()
    if target_text == prediction:
        HIT += 1

acc = HIT / len(df)
print(f"{HIT} among {len(df)}, EM hit acc: {acc}")


