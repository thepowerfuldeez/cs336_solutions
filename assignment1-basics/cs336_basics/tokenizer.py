import math
import pickle
from pathlib import Path

import regex as re
from tqdm.auto import tqdm

from cs336_basics.pretokenization import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = re.compile(PAT)


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        # self.merges: list[tuple[bytes | int, bytes | int]] = []
        self.merges_tuples: list[tuple[bytes, bytes]] = merges
        self.special_tokens = special_tokens

        if special_tokens:
            # longest special token gets matched first
            self.split_re = "(" + "|".join([re.escape(tok) for tok in sorted(self.special_tokens, reverse=True)]) + ")"
        else:
            self.special_tokens = []
            self.split_re = "(" + re.escape("<|endoftext|>") + ")"

        self.vocab: dict[int, bytes] = vocab
        self.byte2int = {v: k for k, v in vocab.items()}

        # add its to special tokens if not available
        for sp in self.special_tokens:
            key = bytes(sp.encode())
            if key not in self.byte2int:
                i = len(self.byte2int[key])
                self.byte2int[key] = i
                self.vocab[i] = key

        self.merges = [
            (self.byte2int[left + right], (self.byte2int[left], self.byte2int[right])) for left, right in merges
        ]

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """Constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens"""
        return cls(
            vocab=pickle.loads(Path(vocab_filepath).read_bytes()),
            merges=pickle.loads(Path(merges_filepath).read_bytes()),
            special_tokens=special_tokens,
        )

    def merge_key(self, left: int | bytes, right: int | bytes, k: tuple, new_id: int) -> tuple[tuple[bytes], bool]:
        """
        Merges a pair of int | bytes into a key and returns a new key
        Example: a1, a2 = (111, 257)
                 k = (100, 111, 257)
                 new_id = 258 -> (100, 258)

        Returns: tuple with updated bytes
        """
        new_k = list(k)
        i = 0
        updated: bool = False
        while i < len(new_k) - 1:
            if new_k[i] == left and new_k[i + 1] == right:
                new_k[i] = new_id
                new_k = new_k[: i + 1] + new_k[i + 2 :]
                updated = True
            i += 1
        return tuple(new_k), updated

    def encode(self, inp: str) -> list[int | bytes]:
        """
        Iteratively apply merges from self.merges to convert a string (sequence of bytes)
        into an encoded sequence
        """
        splitted_by_doc = re.split(self.split_re, inp)
        res = []
        for doc in splitted_by_doc:
            # handle special tokens separately
            if doc in self.special_tokens:
                res.append(self.byte2int[bytes(doc.encode())])
                continue
            for tok in PAT.finditer(doc, concurrent=True):
                tok_bytes = bytes(tok.group().encode())
                key = []
                for i in range(len(tok_bytes)):
                    s = tok_bytes[i : i + 1]
                    if s in self.byte2int:
                        key.append(self.byte2int[s])
                    else:
                        key.extend(list([s]))
                for new_id, (left, right) in self.merges:
                    key, _ = self.merge_key(left, right, key, new_id)
                    if len(key) == 1:
                        break
                res.extend(key)
        return res

    def encode_iterable(self, iterable):
        for line in tqdm(iterable):
            yield from self.encode(line)

    def encode_file(self, inp: str | Path) -> list[int | bytes]:
        file_path = Path(inp)
        chunk_size_in_bytes = 1024 * 1024 * 32
        n_chunks = math.ceil(file_path.stat().st_size / chunk_size_in_bytes)
        with Path(inp).open("rb") as f:
            boundaries = find_chunk_boundaries(f, n_chunks, " ".encode())
            f.seek(0)
            tokens = []
            for start, end in tqdm(list(zip(boundaries[:-1], boundaries[1:]))):
                f.seek(start)
                chunk: str = f.read(end - start).decode("utf-8")
                tokens.extend(self.encode(chunk))
        return tokens

    def decode(self, ids: bytes | int | list[bytes | int]) -> str:
        """
        Returns list of bytes based on the mapping and converts to str

        We need to iteratively expand new tokens so that they are in 0-255 range
        We do that in a while loop and replace initial bytestring with bytes from mapping
        """
        if not isinstance(ids, list):
            ids = [ids]
        x = b""
        for c in ids:
            x += self.vocab.get(c, b"\xff\xfe")
        return x.decode("utf-8", errors="replace")


if __name__ == "__main__":
    tok = Tokenizer.from_files(
        "/Users/george/Projects/learning/assignment1-basics/vocab.pickle",
        "/Users/george/Projects/learning/assignment1-basics/merges.pickle",
        special_tokens=["<|endoftext|>"],
    )
    toks = tok.encode("newest")
    print(toks, tok.decode(toks))

    toks = tok.encode("s")
    print(toks, tok.decode(toks))
