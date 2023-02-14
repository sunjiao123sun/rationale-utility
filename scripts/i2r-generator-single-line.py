from model.simplet5 import SimpleT5
from utils import *
import argparse
import re
import time

time.ctime()
# first i2r then ir2o

parser = argparse.ArgumentParser()
parser.add_argument("--i2r_model", "-m", help="the path for the trained model for the i2r model",
                    default="../model/i2r-cose-1-0-t5-bs-16-epoch-30/simplet5-epoch-29-train-loss-0.0989")
parser.add_argument("--ir2o_model", "-m1", default="../model/cose-1-0-true-t5-bs-16-epoch-30/simplet5-epoch-29-train-loss-0.0066")
parser.add_argument("--test_path", "-t", help="the path for the test data",
                    default="../hf-prepare-i2o/cose-v1-0-True/dev.csv")
parser.add_argument("--gpu", "-g", type=int)
parser.add_argument("--result", "-dr", default="../i2o-hf-test-result/")

args = parser.parse_args()

i2r_model_path = args.i2r_model
ir2o_model_path = args.ir2o_model
test_path = args.test_path

# output_file = open(f"{args.result}/i2r-{i2r_model_path.split('/')[-2]}"
#                    f"-ir2o-{ir2o_model_path.split('/')[-1]}.txt",
#                    "w+")

# output_file = f"{args.result}/i2r-{i2r_model_path.split('/')[-2]}-ir2o-" \
#               f"{ir2o_model_path.split('/')[-1]}-{time.strftime('%l:%M%p %Z on %b %d, %Y')}).csv"
output_file = f"{args.result}/new-train.csv"

i2r_model, ir2o_model = SimpleT5(), SimpleT5()
i2r_model.load_model("t5", i2r_model_path, use_gpu=True, gpu=args.gpu)
ir2o_model.load_model("t5", ir2o_model_path, use_gpu=True, gpu=args.gpu)


df = pd.read_csv(args.test_path)
result = []
for i, row in df.iterrows():
    source_text = row['source_text']
    target_text = row['target_text']
    options_string = re.search('options:(.*)explanation', source_text).group(1).strip()
    if "cose-v1-0" in test_path:
        options = [options_string.split(": ")[1][:-2], options_string.split(": ")[2][:-2],
                   options_string.split(": ")[3]]
    elif "cose-v1-11" in test_path:
        options = [options_string.split(": ")[1][:-2], options_string.split(": ")[2][:-2],
                   options_string.split(": ")[3][:-2], options_string.split(": ")[4][:-2],
                   options_string.split(": ")[5]]
    new_explanation = "explanation:"
    question = source_text.split('  ')[0].replace("context", "explain question")
    # for option in options:
    #     i2r_model_format = f"{question}   answer: {option}"
    #     new_explanation += f"{i2r_model.predict(i2r_model_format)[0].replace('explanation:', '')}."
    answer_string = "\t".join(option for option in options)
    context = f"explain question: {question}   answer: {answer_string}"
    new_explanation += i2r_model.predict(context)[0].replace('explanation:', '')
    print(new_explanation)
    # old_explanation = source_text.split("  ")[-1]
    # if not old_explanation.startswith("explanation"):
    #     raise Exception(f"this example {source_text} does not start with `explanations!!`")
    new_source_text = f"{source_text.split('  ')[0]}  " \
                      f"{source_text.split('  ')[1]}  {new_explanation}"
    print("new source text: ", new_source_text)

    # output_file.write(ir2o_model.predict(new_source_text)[0] + "\n")
    result.append({
        "source_text": source_text,
        "target_text": target_text,
        "generated_rationale": new_explanation,
        "prediction": ir2o_model.predict(new_source_text)[0]
    })

prediction_df = pd.DataFrame(result)
prediction_df.to_csv(output_file, index=False)
preds = prediction_df["prediction"].values

targets = df["target_text"].values
assert len(pd.DataFrame(result)) == len(targets)

HIT = 0
for i in range(0, len(targets)):
    if preds[i].strip() == targets[i].strip():
        HIT += 1

acc = HIT / len(targets)
print(f"{HIT} among {len(targets)}, EM hit acc: {acc}")
# if i > 2:
#     break



