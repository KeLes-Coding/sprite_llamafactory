"""在服务器上统计训练样本经 gemma2 模板编码后的 token 长度（每 N 行抽 1 行）。"""

import json
import sys

from transformers import AutoTokenizer


MODEL = "models/gemma-3-270m-it"
DATA = "data/v3-3-0-3b82a66e/action_sequence_sft_train.jsonl"
CUTOFF = 4096
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 20

tok = AutoTokenizer.from_pretrained(MODEL)


def render(messages):
    system, rest = "", []
    for m in messages:
        if m["role"] == "system":
            system = m["content"] + "\n\n"
        else:
            rest.append(m)
    text = "<bos>"
    for i, m in enumerate(rest):
        role = "user" if m["role"] == "user" else "model"
        content = (system + m["content"]) if i == 0 else m["content"]
        text += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
    return text


lengths = []
with open(DATA, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i % STEP:
            continue
        lengths.append(len(tok(render(json.loads(line)["messages"]), add_special_tokens=False)["input_ids"]))

lengths.sort()
n = len(lengths)
print(f"sampled={n} p50={lengths[n // 2]} p90={lengths[int(n * 0.9)]} p99={lengths[int(n * 0.99)]} max={lengths[-1]}")
print(f"over_cutoff({CUTOFF})={sum(x > CUTOFF for x in lengths)}  mean={sum(lengths) / n:.0f}")
