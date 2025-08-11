import math
import pickle
import time
import logging
from pathlib import Path
from itertools import chain, count
from dataclasses import dataclass
from functools import lru_cache

import regex as re
from tqdm.auto import tqdm

from cs336_basics.pretokenization import find_chunk_boundaries

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


@dataclass()
class Token:
    value: int
    next = None
    prev = None
    version: int = 0

    def __repr__(self):
        if self.next is not None:
            n = self.next.value
        else:
            n = None
        if self.prev is not None:
            p = self.prev.value
        else:
            p = None
        if p is None and n is None:
            return "X"
        return f"Token({self.value}, v={self.version}, n={n}, p={p})"


class MinHeap:
    def __init__(self, bpe_ranks: dict[tuple[int, int], int]):
        self.heap = []
        heapq.heapify(self.heap)
        self.tie_breaker = count()
        self.bpe_ranks = bpe_ranks

    def push(self, left: Token):
        if left and left.next:
            pair = (left.value, left.next.value)
            # logger.info(f"check {pair}")
            if pair in self.bpe_ranks:
                rank = self.bpe_ranks[pair]
                tie_breaker = next(self.tie_breaker)
                # logger.info(f"push {(rank, tie_breaker, left.version, left, left.next)}")
                heapq.heappush(self.heap, (rank, tie_breaker, left.version, left))

    def pop(self):
        rank, _, version_at_push, node = heapq.heappop(self.heap)
        # logger.info(f"pop {rank=} {version_at_push=} {node=}")
        return rank, node, version_at_push

    def __bool__(self):
        return bool(self.heap)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def process(index_and_args) -> tuple[int, list[int]]:
#     index, args = index_and_args
#     (f, start, end, split_re, special_tokens, byte2int, PAT, merges, merge_key) = args
#     chunk: str = f.read(end - start).decode("utf-8")

#     res = []
#     for doc in re.splititer(split_re, chunk, concurrent=True):
#         # handle special tokens separately
#         if doc in special_tokens:
#             res.append(byte2int[bytes(doc.encode())])
#             continue
#         for tok in PAT.finditer(doc, concurrent=True):
#             tok_bytes = bytes(tok.group().encode())
#             key = []
#             for i in range(len(tok_bytes)):
#                 s = tok_bytes[i : i + 1]
#                 if s in byte2int:
#                     key.append(byte2int[s])
#                 else:
#                     key.extend(list([s]))
#             for new_id, (left, right) in merges:
#                 key, _ = merge_key(left, right, key, new_id)
#                 if len(key) == 1:
#                     break
#             res.extend(key)
#     return index, res

import numpy as np
import heapq


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        # Special tokens and split regex
        self.special_tokens = special_tokens or []
        # Vocabulary and reverse map
        self.vocab = vocab
        self.byte2int = {v: k for k, v in vocab.items()}
        self.lut256 = [-1] * 256
        for b in range(256):
            self.lut256[b] = self.byte2int[bytes([b])]

        # self.merges = [
        #     ((self.byte2int[left], self.byte2int[right]), self.byte2int[left + right]) for left, right in merges
        # ]

        escaped = [re.escape(tok) for tok in sorted(self.special_tokens, reverse=True)]
        base_tok = "<|endoftext|>"
        self.split_re = f"({'|'.join(escaped)})" if escaped else f"({re.escape(base_tok)})"

        # Build merge_map: (left_id, right_id) -> new_id
        self.merge_map: dict[tuple[int, int], int] = {}
        self.bpe_ranks: dict[tuple[int, int], int] = {}
        for rank, (left_bytes, right_bytes) in enumerate(merges):
            # IDs for left and right tokens
            left_id = self.byte2int[left_bytes]
            right_id = self.byte2int[right_bytes]
            # combined bytes sequence should already be in vocab for BPE merges
            combined = left_bytes + right_bytes
            new_id = self.byte2int.get(combined)
            if new_id is None:
                # fallback: map concatenation not in vocab
                continue
            pair = (left_id, right_id)
            self.merge_map[pair] = new_id
            self.bpe_ranks[pair] = rank

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

    @lru_cache(maxsize=100_000)
    def _encode_bytes_cached(self, tok_bytes: bytes) -> tuple[int]:
        key = [self.lut256[b] for b in tok_bytes]
        return tuple(self._heap_merge(key))

    def _heap_merge(self, key: list[int | bytes]) -> list[int]:
        """
        1. Build a double linked list from input sequence
            Each node has value, version, next and prev
        2. Use MinHeap with tie breaker as itertools.count
        3. At every iteration, take a node with lowest bpe rank
            then validate it has right neighbor and collapse to new id
            Increase version of left node
            Relink left node to a new neighbor (left.next.next)
        4. Add 2 neighbors after merge back to the heap
        """
        tokens = [Token(v) for v in key]
        heap = MinHeap(bpe_ranks=self.bpe_ranks)

        for i in range(len(tokens) - 1):
            tokens[i].next = tokens[i + 1]
            tokens[i + 1].prev = tokens[i]
        # logger.info("---------------")
        # logger.info(tokens)
        # logger.info([self.decode(n.value) for n in tokens])

        for n in tokens[:-1]:
            heap.push(n)

        while heap:
            rank_at_push, left, version_at_push = heap.pop()
            if left.version != version_at_push:
                # logger.info("version mismatch")
                continue
            right = left.next
            if not right:
                # logger.info("no right node")
                continue
            pair = (left.value, right.value)
            if pair not in self.bpe_ranks:
                # logger.info("incorrect pair")
                continue
            if self.bpe_ranks[pair] != rank_at_push:
                # logger.info(f"incorrect rank: {self.bpe_ranks[pair]=} {rank_at_push=}")
                continue

            new_id = self.merge_map[pair]
            left.value = new_id
            left.version += 1
            # logger.info(f"{pair} -> {new_id}")

            # unlink right (fully detach and invalidate)
            rn = right.next
            right.next = None
            right.prev = None
            right.version += 1

            # relink left
            left.next = rn
            if rn:
                rn.prev = left

            # logger.info(tokens)

            if left.prev:
                heap.push(left.prev)
            heap.push(left)
        out = []
        cur = tokens[0]
        # rollback to find actual head
        while cur and cur.prev:
            cur = cur.prev
        while cur:
            out.append(cur.value)
            cur = cur.next
        # logger.info(out)
        return out

    def encode(self, inp: str) -> list[int | bytes]:
        out: list[int | bytes] = []
        for doc in re.splititer(self.split_re, inp, concurrent=True):
            if doc and doc in self.special_tokens:
                token_bytes = doc.encode()
                out.append(self.byte2int[token_bytes])
                continue
            for i, tok in enumerate(PAT.finditer(doc, concurrent=True)):
                tok_bytes = tok.group().encode("utf-8")
                key = self._encode_bytes_cached(tok_bytes)
                out.extend(key)
        return out

    def encode_iterable(self, iterable):
        for line in tqdm(iterable, desc="Lines processed: "):
            yield from self.encode(line)

    def encode_file(self, filepath: str | Path, chunk_size: int = 1024 * 1024) -> list[int]:
        path = Path(filepath)
        size = path.stat().st_size
        n_chunks = math.ceil(size / chunk_size)
        boundaries = find_chunk_boundaries(path.open("rb"), n_chunks, b" ")
        tokens: list[int] = []
        with path.open("rb") as f:
            for start, end in tqdm(zip(boundaries[:-1], boundaries[1:]), total=len(boundaries) - 1):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                tokens.extend(self.encode(chunk))
        return tokens

    def decode(self, ids: int | bytes | list[int | bytes]) -> str:
        if not isinstance(ids, list):
            ids = [ids]
        b = b""
        for i in ids:
            if isinstance(i, int):
                b += self.vocab.get(i, b"")
            else:
                b += i
        return b.decode("utf-8", errors="replace")


if __name__ == "__main__":
    tok = Tokenizer.from_files(
        "/Users/george/Projects/learning/assignment1-basics/vocab.pickle",
        "/Users/george/Projects/learning/assignment1-basics/merges.pickle",
        special_tokens=["<|endoftext|>"],
    )
    input_dir = Path("/Users/george/Projects/learning/assignment1-basics/data/")

    # toks = tok.encode("newest")
    # print(toks, tok.decode(toks))

    # toks = tok.encode("s")
    # print(toks, tok.decode(toks))

    # print(max(list(tok.vocab.items()), key=lambda x: len(x[1])))

    # # sample 10 docs from tinystories
    # ratios = []
    # for doc in re.split(tok.split_re, (input_dir / "TinyStoriesV2-GPT4-valid.txt").read_text())[:20]:
    #     if doc != "<|endoftext|>":
    #         n_bytes = len(doc.encode())
    #         print(doc)
    #         toks = tok.encode(doc)
    #         print(toks)
    #         n_tokens = len(toks)
    #         ratio = n_bytes / n_tokens
    #         print(f"compression ratio: {ratio}")
    #         ratios.append(ratio)

    # import numpy as np
    # print(np.mean(ratios))

    tokenized_path = Path("data_tokenized")
    tokenized_path.mkdir(exist_ok=True, parents=True)

    # for fname in ["TinyStoriesV2-GPT4-valid.txt", "TinyStoriesV2-GPT4-train.txt"]:
    #     t0 = time.monotonic()
    #     tokens = tok.encode_file(input_dir / fname, chunk_size=8 * 1024 * 1024)
    #     taken = time.monotonic() - t0
    #     logger.info(f"Took {taken:.1f} s.")
    #     logger.info(f"Throughput: {22 / taken:.2f} MB/s")
    #     np.save(str((tokenized_path / fname).with_suffix(".npy")), np.array(tokens, dtype="uint16"))


    tok = Tokenizer.from_files(
        "/Users/george/Projects/learning/assignment1-basics/vocab_owt.pickle",
        "/Users/george/Projects/learning/assignment1-basics/merges_owt.pickle",
        special_tokens=["<|endoftext|>"],
    )
    for fname in ["owt_valid.txt", "owt_train.txt"]:
        t0 = time.monotonic()
        fpath = input_dir / fname
        tokens = tok.encode_file(fpath, chunk_size=8 * 1024 * 1024)
        taken = time.monotonic() - t0
        logger.info(f"Took {taken:.1f} s.")
        logger.info(f"Throughput: {fpath.stat().st_size / (1024 * 1024) / taken:.2f} MB/s")
        np.save(str((tokenized_path / fname).with_suffix(".npy")), np.array(tokens, dtype="uint16"))