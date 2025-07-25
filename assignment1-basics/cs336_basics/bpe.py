import logging
import os
import math
import pickle
import time
from pathlib import Path
from collections import defaultdict

import regex as re
from tqdm.auto import tqdm

from cs336_basics.pretokenization import Splitter, pre_tokenize, find_chunk_boundaries

logging.basicConfig(level=os.getenv("LOGLEVEL", logging.INFO))
logger = logging.getLogger(__name__)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = re.compile(PAT)


class BPE:
    def __init__(self, special_tokens: list[str], vocab_size: int = 355):
        self.vocab_size: int = vocab_size

        self.merges: list[tuple[bytes | int, bytes | int]] = []
        self.merges_tuples: list[tuple[bytes, bytes]] = []
        self.special_tokens = special_tokens
        self.special_tokens_bytes = [x.encode() for x in special_tokens]

        self.splitter = Splitter("<|endoftext|>")
        self.split_re = "(" + "|".join([re.escape(tok) for tok in self.special_tokens]) + ")"

        self.vocab: dict[int, bytes] = {256 + i: special_tok for i, special_tok in enumerate(self.special_tokens_bytes)}
        self.new_id_to_bytes: dict[int, int | bytes] = self.vocab.copy()
        for i in range(256):
            self.vocab[i] = bytes([i])

        self.sort_time = 0
        self.second_best_key = None

    @property
    def cur_vocab_size(self):
        """Current vocab size (different from self.vocab_size which is target vocab size)"""
        return len(self.vocab)

    def break_ties(self, sorted_all_counts):
        """
        This can be replaced with single sort_key function that is passed as a key to sorted
        but it would be a bit slower
        """
        _, max_cnt = sorted_all_counts[0]
        ties = []
        candidates = []
        for i in range(len(sorted_all_counts)):
            key, count = sorted_all_counts[i]
            if count == max_cnt:
                ties.append(tuple(self.vocab[id_] for id_ in key))
                candidates.append(key)
            else:
                break
        best_from_ties = max(ties)
        for i, cand in enumerate(candidates):
            if ties[i] == best_from_ties:
                max_key = cand
                break
        return max_key

    def convert(self, entry: bytes | int) -> bytes:
        """
        Convert tokens
        """
        if entry in self.vocab:
            return self.vocab[entry]
        bytestr: list[bytes] = self.new_id_to_bytes.get(entry, [entry])
        while any(elem in self.new_id_to_bytes for elem in bytestr):
            i = 0
            while i < len(bytestr):
                if bytestr[i] in self.new_id_to_bytes:
                    bytestr = bytestr[:i] + self.new_id_to_bytes[bytestr[i]] + bytestr[i + 1 :]
                i += 1
        else:
            return bytes(bytestr)

    def decode(self, entry: bytes | int | list[bytes | int]) -> str:
        """
        Returns list of bytes based on the mapping and converts to str

        We need to iteratively expand new tokens so that they are in 0-255 range
        We do that in a while loop and replace initial bytestring with bytes from mapping
        """
        if isinstance(entry, list) or isinstance(entry, tuple):
            return "".join([self.convert(e).decode("utf-8", errors="replace") for e in entry])
        return self.convert(entry).decode("utf-8", errors="replace")

    def encode(self, inp: str) -> list[int | bytes]:
        """
        Iteratively apply merges from self.merges to convert a string (sequence of bytes)
        into an encoded sequence
        """
        splitted_by_doc = re.split(self.split_re, inp)
        res = []
        for doc in splitted_by_doc:
            if doc in self.special_tokens:
                res.append(256 + self.special_tokens.index(doc))
                continue
            for tok in PAT.finditer(doc, concurrent=True):
                key = list(tok.group().encode())
                for new_id, (left, right) in enumerate(self.merges, 256 + len(self.special_tokens)):
                    key, _ = self.merge_key(left, right, key, new_id)
                    if len(key) == 1:
                        break
                res.extend(key)
        return res

    def encode_file(self, inp: str | Path) -> list[int | bytes]:
        file_path = Path(inp)
        chunk_size_in_bytes = 1024 * 1024 * 32
        n_chunks = math.ceil(file_path.stat().st_size / chunk_size_in_bytes)
        with Path(inp).open("rb") as f:
            boundaries = find_chunk_boundaries(f, n_chunks, " ".encode())
            f.seek(0)
            tokens = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk: str = f.read(end - start).decode("utf-8")
                tokens.extend(self.encode(chunk))
        return tokens

    def update_counts(
        self,
        pre_token_byte_counts: dict[tuple, int],
        pair_to_pre_tokens: dict[tuple, set] | None,
        all_counts: dict | None = None,
    ):
        """
        Update counts of all_counts by v with each pair of k
        Also update pair_to_pre_tokens if passed
        """
        if all_counts is None:
            all_counts: dict[tuple[bytes], int] = defaultdict(int)
        else:
            all_counts = defaultdict(int, all_counts)
        for k, v in pre_token_byte_counts.items():
            for i in range(len(k) - 1):
                key = (k[i], k[i + 1])
                all_counts[key] += v
                if pair_to_pre_tokens is not None:
                    pair_to_pre_tokens.setdefault(key, set()).add(k)
        return all_counts

    def iter_merge_cached(
        self,
        pre_token_byte_counts: dict[tuple[bytes], int],
        updated_keys=None,
        all_counts=None,
        pair_to_pre_tokens=None,
        all_updated_pairs=None,
    ) -> tuple[tuple[tuple[bytes], int], dict, set, dict, dict, set]:
        """
        More efficient implementation of iter_merge, that works 4x faster on 1000 iters (1.7s vs 6.3s)

        We operate on extra knowledge that we are only interested in updated pairs in iteration i-1
        at iteration i

        1. If we don't pass cached updated_keys, all_counts, pair_to_pre_tokens then algo is the same as iter_merge
        2. Instead, we just iterate over provided updated_keys (instead of all pre-tokens) and update provided
        all_counts
        3. Find rough estimation of pre-tokens by finding intersection with most frequent pair
        obtaining affected_pre_tokens
        4. Update pre_token_byte_counts more smartly -- only operating across affected_pre_tokens, where likely
        pre-tokens would change after the merge_key function
        5. Subtract count from every pair in a key that gets updated, and keep pair_to_pre_tokens updated
        At next iteration, only required keys will be updated and all of their pairs will be used
        """
        if pair_to_pre_tokens is None:
            # pair -> count
            pair_to_pre_tokens = {}
            all_counts: dict[tuple[bytes | int], int] = self.update_counts(pre_token_byte_counts, pair_to_pre_tokens)
            all_updated_pairs = set(all_counts.keys())
        else:
            # all_counts, pair_to_pre_tokens, pre_tokens_to_pairs, all_updated_pairs are restored from args
            all_counts = self.update_counts(
                {k: pre_token_byte_counts[k] for k in updated_keys}, pair_to_pre_tokens, all_counts=all_counts
            )

        # identify the most frequent pair
        tx = time.monotonic()

        # Optimized sorting algorithm -- keep checking only updated pairs + all previous sorted subset
        if self.second_best_key is not None and all_updated_pairs:
            sorted_subset = sorted(
                [
                    (k, all_counts[k])
                    for k in set([k for (k, v) in self.second_best_key]).union(all_updated_pairs)
                    if k in all_counts
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            self.second_best_key = sorted_subset
            max_key = self.break_ties(sorted_subset)
        else:
            sorted_all_counts = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
            # We keep top10% of current iteration and re-use only that in next iter during sorting 
            # This shaves time dramatically
            count_to_keep = math.ceil(len(sorted_all_counts) * 0.10)
            self.second_best_key = sorted_all_counts[:count_to_keep]
            max_key = self.break_ties(sorted_all_counts)

        tx1 = time.monotonic()
        self.sort_time += tx1 - tx
        new_id = self.cur_vocab_size
        # logger.info(f"max freq is {all_counts[max_key]=}, max_key={self.decode(max_key)}")

        affected_pre_tokens: set[tuple[bytes]] = set()
        for (left, right), keys in pair_to_pre_tokens.items():
            if left in max_key and right in max_key:
                affected_pre_tokens.update(keys)

        new_pre_token_byte_counts = pre_token_byte_counts.copy()
        new_updated_keys: set[tuple[bytes]] = set()

        all_counts_updated = all_counts.copy()
        pair_to_pre_tokens_updated = pair_to_pre_tokens.copy()
        all_updated_pairs_updated = set()

        left, right = max_key
        for k in affected_pre_tokens:
            new_k, updated_indices = self.merge_key(left, right, k, new_id)
            if updated_indices:
                v = new_pre_token_byte_counts.pop(k)
                for pair in zip(k[:-1], k[1:]):
                    all_counts_updated[tuple(pair)] -= v
                    assert all_counts_updated[tuple(pair)] >= 0
                    pair_to_pre_tokens_updated[tuple(pair)].discard(k)

                # counts are not changed, we just need to re-write with a new key
                new_pre_token_byte_counts[new_k] = v
                new_updated_keys.add(new_k)

                for j in updated_indices:
                    if j > 0:
                        all_updated_pairs_updated.add((new_k[j - 1], new_k[j]))
                    if j < len(new_k) - 1:
                        all_updated_pairs_updated.add((new_k[j], new_k[j + 1]))

        return (
            (max_key, new_id),
            new_pre_token_byte_counts,
            new_updated_keys,
            {k: v for k, v in all_counts_updated.items() if v > 0},
            pair_to_pre_tokens_updated,
            all_updated_pairs_updated,
        )

    def iter_merge(
        self,
        pre_token_byte_counts: dict[tuple[bytes], int],
    ) -> tuple[tuple[tuple[bytes], int], dict]:
        """
        More efficient implementation of iter_merge, that works 4x faster on 1000 iters (1.7s vs 6.3s)

        We operate on extra knowledge that we are only interested in updated pairs in iteration i-1
        at iteration i

        1. If we don't pass cached updated_keys, all_counts, pair_to_pre_tokens then algo is the same as iter_merge
        2. Instead, we just iterate over provided updated_keys (instead of all pre-tokens) and update provided
        all_counts
        3. Find rough estimation of pre-tokens by finding intersection with most frequent pair
        obtaining affected_pre_tokens
        4. Update pre_token_byte_counts more smartly -- only operating across affected_pre_tokens, where likely
        pre-tokens would change after the merge_key function
        5. Subtract count from every pair in a key that gets updated, and keep pair_to_pre_tokens updated
        At next iteration, only required keys will be updated and all of their pairs will be used
        """
        # pair -> count
        all_counts: dict[tuple[bytes], int] = defaultdict(int)
        for k, v in pre_token_byte_counts.items():
            for i in range(len(k) - 1):
                all_counts[(k[i], k[i + 1])] += v

        # identify the most frequent pair
        tx = time.monotonic()

        sorted_all_counts = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
        max_key = self.break_ties(sorted_all_counts)

        tx1 = time.monotonic()
        self.sort_time += tx1 - tx
        if (tx1 - tx) > 0.01:
            logger.info(f"sort time is too long, took {tx1 - tx:.02f} s.")
        new_id = self.cur_vocab_size

        new_pre_token_byte_counts = pre_token_byte_counts.copy()

        left, right = max_key
        for k, v in pre_token_byte_counts.items():
            new_k, updated = self.merge_key(left, right, k, new_id)
            if updated:
                v = new_pre_token_byte_counts.pop(k)
                # counts are not changed, we just need to re-write with a new key
                new_pre_token_byte_counts[new_k] = v

        return (
            (max_key, new_id),
            new_pre_token_byte_counts,
        )

    def merge_key(self, left: int | bytes, right: int | bytes, k: tuple, new_id: int) -> tuple[tuple[bytes], list]:
        """
        Merges a pair of int | bytes into a key and returns a new key
        Example: a1, a2 = (111, 257)
                 k = (100, 111, 257)
                 new_id = 258 -> (100, 258)

        Returns: tuple with updated bytes
        """
        new_k = list(k)
        i = 0
        # updated indices in the modified key
        updated_indices: list[int] = []

        while i < len(new_k) - 1:
            if new_k[i] == left and new_k[i + 1] == right:
                new_k[i] = new_id
                new_k = new_k[: i + 1] + new_k[i + 2 :]
                updated_indices.append(i)
            i += 1
        return tuple(new_k), updated_indices

    def train(self, filepath: str, num_processes: int = 1):
        logger.info("Starting to train BPE")
        t0 = time.monotonic()
        pre_token_counts = pre_tokenize(self.splitter, str(filepath), num_processes=num_processes)

        t1 = time.monotonic()
        logger.info(f"Pre-tokenization finished in {t1 - t0:.1f} s.")
        self.pre_token_byte_counts: dict[tuple[bytes], int] = {
            tuple(k.encode()): v for k, v in pre_token_counts.items()
        }

        n_iters = max(0, self.vocab_size - self.cur_vocab_size)
        logger.info(f"Using {n_iters=}")

        # Non-efficient implementation
        # for i in range(n_iters):
        #     (updated_key, new_id), self.pre_token_byte_counts = self.iter_merge(self.pre_token_byte_counts)
        #     self.new_id_to_bytes[new_id] = updated_key
        #     v = self.convert(new_id)
        #     self.merges.append(updated_key)
        #     converted = (self.convert(updated_key[0]), self.convert(updated_key[1]))
        #     merges_tuples.append(converted)
        #     self.vocab[new_id] = v
        #     logger.info(f"iter: {i}, updated new id mapping with {new_id=}, {v=}")

        # cached, more efficient version
        updated_keys, all_counts, pair_to_pre_tokens, all_updated_pairs = None, None, None, None
        bar = tqdm(total=n_iters, desc="Training BPE")
        for i in range(n_iters):
            (
                (updated_key, new_id),
                self.pre_token_byte_counts,
                new_updated_keys,
                all_counts_updated,
                pair_to_pre_tokens_updated,
                all_updated_pairs_updated,
            ) = self.iter_merge_cached(
                self.pre_token_byte_counts, updated_keys, all_counts, pair_to_pre_tokens, all_updated_pairs
            )
            self.new_id_to_bytes[new_id] = updated_key
            v = self.convert(new_id)
            self.merges.append(updated_key)
            converted = (self.convert(updated_key[0]), self.convert(updated_key[1]))
            # logger.info(f"merges at step {i}: {converted}")
            self.merges_tuples.append(converted)
            self.vocab[new_id] = v
            # logger.info(f"iter: {i}, updated new id mapping with {new_id=}, {v=}")

            updated_keys, all_counts, pair_to_pre_tokens, all_updated_pairs = (
                new_updated_keys,
                all_counts_updated,
                pair_to_pre_tokens_updated,
                all_updated_pairs_updated,
            )
            bar.update()
        t2 = time.monotonic()
        logger.info(f"Finished training in {t2 - t0:.1f} s.\nAverage iter time: {(t1 - t0) / n_iters:.5f} s.")
        logger.info(f"Total sort time was {self.sort_time:.2f} s.")
        return self.vocab, self.merges_tuples

    def save(self, vocab_path: str, merges_path: str):
        Path(vocab_path).write_bytes(pickle.dumps(self.vocab))
        Path(merges_path).write_bytes(pickle.dumps(self.merges_tuples))


if __name__ == "__main__":
    filepath = "data/owt_train.txt"
    # filepath = "data/TinyStoriesV2-GPT4-train.txt"
    # filepath = "data/TinyStoriesV2-GPT4-mid3.txt"
    # filepath = "data/TinyStoriesV2-GPT4-valid.txt"
    # filepath = "cs336_basics/test.txt"
    # filepath = "tests/fixtures/tinystories_sample_5M.txt"
    bpe = BPE(["<|endoftext|>"], vocab_size=32000)
    # bpe = BPE(["<|endoftext|>"], vocab_size=10000)
    # bpe = BPE(["<|endoftext|>"], vocab_size=1000)
    vocab, merges = bpe.train(filepath, num_processes=8)

    logger.info([bpe.decode(x) for x in list(vocab)[256:356]])
    toks = bpe.encode("newest is a newest")
    logger.info(toks)
    logger.info([bpe.decode(tok) for tok in toks])

    bpe.save("vocab_owt.pickle", "merges_owt.pickle")
