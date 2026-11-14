"""Score and stably rank query-document pairs with Qwen yes/no rerankers."""

import logging
import math


INSTRUCTION = "Given a query, retrieve relevant documents that answer the query"
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MODEL_IDS = {
    "0.6B": "Qwen/Qwen3-Reranker-0.6B",
    "4B": "Qwen/Qwen3-Reranker-4B",
    "8B": "Qwen/Qwen3-Reranker-8B",
}


def prompt(query, document, instruction=INSTRUCTION):
    """Return a complete yes/no chat prompt from query and document strings."""
    return PREFIX + f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}" + SUFFIX


class QwenReranker:
    """Batch query-document scoring with a 2048-token input limit.

    Scores are yes probabilities normalized over the no/yes logits, rather than
    generated answers. CUDA devices use FP16; other devices use FP32. Pair and
    truncation counts accumulate until close() logs them and releases the model.
    """

    def __init__(self, model_id, device="cuda:0", batch_size=32):
        """Load a model/tokenizer ID on device with a positive pair batch size."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_length = 2048
        self.pairs = self.truncated_pairs = 0
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        no = self.tokenizer.encode("no", add_special_tokens=False)
        yes = self.tokenizer.encode("yes", add_special_tokens=False)
        if len(no) != 1 or len(yes) != 1 or no == yes:
            raise ValueError("Expected distinct single-token yes/no answers")
        self.answer_ids = [no[0], yes[0]]
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def score(self, query, documents, instruction=INSTRUCTION):
        """Return one finite yes probability per document, preserving input order.

        Inputs are strings. Prompts are truncated to the token limit; an empty
        document list returns []. Pair counts include truncated prompts.
        """
        scores = []
        for start in range(0, len(documents), self.batch_size):
            texts = [prompt(query, doc, instruction) for doc in documents[start:start + self.batch_size]]
            lengths = self.tokenizer(texts, truncation=False, return_length=True)["length"]
            self.pairs += len(texts)
            self.truncated_pairs += sum(n > self.max_length for n in lengths)
            encoded = self.tokenizer(texts, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                # Normalize only the no/yes logits at the final input position.
                logits = self.model(**encoded).logits[:, -1, self.answer_ids].float()
                scores.extend(logits.log_softmax(dim=1)[:, 1].exp().tolist())
        if any(not math.isfinite(s) for s in scores):
            raise ValueError("Non-finite reranker score")
        return scores

    def rank(self, query, candidates, documents, instruction=INSTRUCTION):
        """Return candidate IDs by descending score, breaking ties by input order.

        Documents maps every candidate ID to text. No candidates are added or
        removed, and the supplied order and document mapping are left unchanged.
        """
        scores = self.score(query, [documents[c] for c in candidates], instruction)
        return [candidates[i] for i in sorted(range(len(candidates)), key=lambda i: (-scores[i], i))]

    def close(self):
        """Log pair counts and release the model and unused CUDA cache once."""
        logging.info("pairs=%d truncated_pairs=%d", self.pairs, self.truncated_pairs)
        del self.model
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
