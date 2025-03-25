import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm, trange

import faiss

from ..data import get_template_and_fix_tokenizer
from ..extras.constants import CHOICES, SUBJECTS
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from .template import get_eval_template

if TYPE_CHECKING:
    from numpy.typing import NDArray

class IndexBuilder:
    def __init__(self, args: Dict[str, Any]) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "right" 
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.tokenizer, self.model_args, finetuning_args)
        self.eval_template = get_eval_template(self.eval_args.lang)
        self.choice_inputs = [self.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in CHOICES]

    @torch.inference_mode()
    def batch_inference(self, batch_input: Dict[str, "torch.Tensor"]) -> List[str]:
        logits = self.model(**batch_input).logits
        lengths = torch.sum(batch_input["attention_mask"], dim=-1)
        word_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
        choice_probs = torch.nn.functional.softmax(word_probs[:, self.choice_inputs], dim=-1).detach()
        return [chr(ord("A") + offset.item()) for offset in torch.argmax(choice_probs, dim=-1)]

    def build(self) -> None:
        dataset = load_dataset("llamafactory/OpenR1-Math-94k", split="train")
        limit = 200
        dataset = dataset.shuffle(seed=42).select(range(limit))
        inputs, outputs, labels = [], [], []
        for i in trange(len(dataset), desc="preparing data"):
            messages = self.eval_template.format_example(
                target_data=dataset[i],
                support_set=(),
            )