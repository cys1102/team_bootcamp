import pandas as pd
import torch
from kobart import get_kobart_tokenizer
from transformers import PreTrainedTokenizerFast
from transformers.models.bart import BartForConditionalGeneration


test_data = pd.read_csv("open/test.csv", engine="python", error_bad_lines=False)
sample_submission = pd.read_csv(
    "open/sample_submission.csv", engine="python", error_bad_lines=False
)
test_text = test_data["text"]
summary = []

model = BartForConditionalGeneration.from_pretrained("kobart_summary")
# tokenizer = get_kobart_tokenizer()
tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-summarization")


def text2summary(text):
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output


for i in range(len(test_text)):
    summary.append(text2summary(test_text[i]))

sample_submission["summary"] = summary
sample_submission.to_csv("submission.csv", index=False)
