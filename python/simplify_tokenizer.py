'''
来自https://github.com/taifyang/sam-inference/blob/main/sam3/simplify_tokenizer.py
modelscope download --model facebook/sam3 --local_dir /data/sunkx/sam3/model_sam3
'''
import json
import regex as re
from typing import Dict, List, Tuple, Optional


def bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: Tuple[str, ...]):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class SimpleCLIPBPETokenizer:
    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        max_length: int = 32,
        bos_token_id: int = 49406,
        eos_token_id: int = 49407,
        added_tokens: Optional[Dict[str, int]] = None,
        bpe_vocab_size: int = 49152,  
        do_lower_case: bool = True,
    ) -> None:
        self.max_length = max_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.do_lower_case = do_lower_case

        if added_tokens is None:
            added_tokens = {
                "<|startoftext|>": bos_token_id,
                "<|endoftext|>": eos_token_id,
            }
        self.added_tokens: Dict[str, int] = added_tokens
        self.unk_token: str = "<|endoftext|>"

        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"""
            r"""[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

        self.byte_encoder = bytes_to_unicode()

        with open(merges_file, encoding="utf-8") as f:
            merges = f.read().strip().split("\n")[1 : bpe_vocab_size - 256 - 2 + 1]
        merges_pairs = [tuple(m.split()) for m in merges]
        self.bpe_ranks: Dict[Tuple[str, str], int] = {
            pair: i for i, pair in enumerate(merges_pairs)
        }

        with open(vocab_file, encoding="utf-8") as f:
            self.encoder: Dict[str, int] = json.load(f)

    def _bpe(self, token: str) -> str:
        bpe_ranks = self.bpe_ranks

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: bpe_ranks.get(pair, float("inf")))
            if bigram not in bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if (
                    word[i] == first
                    and i < len(word) - 1
                    and word[i + 1] == second
                ):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        return " ".join(word)

    def _tokenize(self, text: str) -> List[str]:
        if self.do_lower_case:
            text = text.lower()

        bpe_tokens: List[str] = []

        for token in self.pat.findall(text):
            # bytes -> unicode string
            token_encoded = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )
            bpe_out = self._bpe(token_encoded)
            bpe_tokens.extend(bpe_out.split(" "))

        return bpe_tokens

    def _token_to_id(self, token: str) -> int:
        if token in self.added_tokens:
            return self.added_tokens[token]
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def encode(self, text: str) -> Tuple[List[int], List[int]]:
        bpe_tokens = self._tokenize(text)

        ids = [self._token_to_id(tok) for tok in bpe_tokens]
        len_ids = len(ids)

        num_special = 2  # BOS + EOS
        total_len = len_ids + num_special

        ids = [self.bos_token_id] + ids + [self.eos_token_id]

        ids = ids[: self.max_length]

        valid_len = min(total_len, self.max_length)

        if len(ids) < self.max_length:
            ids = ids + [self.eos_token_id] * (self.max_length - len(ids))

        attention_mask = [1] * valid_len + [0] * (self.max_length - valid_len)

        return ids, attention_mask


if __name__ == "__main__":
    vocab_file = "./model_sam3/vocab.json"
    merges_file = "./model_sam3/merges.txt"

    tokenizer = SimpleCLIPBPETokenizer(
        vocab_file=vocab_file,
        merges_file=merges_file,
        max_length=32,
        bos_token_id=49406,
        eos_token_id=49407,
        bpe_vocab_size=49152,
    )

    prompt = "person"
    ids, mask = tokenizer.encode(prompt)
    print("ids:", ids)
    print("len(ids):", len(ids))
    print("mask:", mask)
    print("sum(mask):", sum(mask))