import os
import numpy as np
import pandas as pd
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Args:
    batch_size = 64
    max_len = 512


class TestDataset(Dataset):
    def __init__(self, tok, data, max_len, pad_index=0):
        super().__init__()
        self.tok = tok
        self.data = data
        self.max_len = max_len
        self.len = len(self.data)
        self.pad_index = pad_index

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[: self.max_len]

        return inputs

    def __getitem__(self, idx):
        input_ids = self.tok.encode(self.data[idx])
        input_ids = self.add_padding_data(input_ids)
        return np.array(input_ids, dtype=np.int_)

    def __len__(self):
        return self.len


def text2input(text, tokenizer):
    return tokenizer.encode(text, return_tensors="pt").unsqueeze(0)


def setup():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_data = pd.read_csv("open/test_data.csv", engine="python", error_bad_lines=False)
    sample_submission = pd.read_csv(
        "open/sample_submission.csv", engine="python", error_bad_lines=False
    )
    test_text = test_data["text"]

    model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-summarization").to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-summarization")

    return device, model, tokenizer, sample_submission, test_text


if __name__ == "__main__":
    args = Args()
    device, model, tokenizer, sample_submission, test_text = setup()
    summary = []

    test_dataset = TestDataset(tokenizer, test_text, args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    for (idx, batch) in enumerate(tqdm(test_dataloader)):
        pred = model.generate(batch.to(device), eos_token_id=1, max_length=128, num_beams=5)
        for i in range(len(pred)):
            summary.append(
                tokenizer.decode(
                    pred[i].detach().cpu().squeeze().numpy(),
                    skip_special_tokens=True,
                )
            )
    sample_submission["summary"] = summary
    sample_submission.to_csv("submission.csv", index=False, encoding="utf-8")
