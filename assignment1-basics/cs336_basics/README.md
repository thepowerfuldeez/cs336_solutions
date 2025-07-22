### Speed profiling
```bash
sudo py-spy record -f speedscope -o profile.json -- python bpe1.py
```

### Pre-compiling fastsplit
```
curl https://sh.rustup.rs -sSf | sh
```

```
cd fastsplit
maturin develop --release
```




####
Optimization

I re-wrote part with pre-tokenization which splits data by regexp and counts pre-tokens
Before it was 67.2s on 1.5M lines of text, in which 45.6s took by split, 17.3s took by get_counts

Iter 1: 26s for split, and same 17.3s for get_counts

Iter 2: 21.2s total time (split + get counts)

Iter 2 is much faster because count is now happens inside rust code, and no large list: str gets marshalled via pickle from pyo3

Using multiple processes works best on a large file, so
1 process 77.2s
2 processes 73.5s
8 processes 69.2s

(this can be tested via running `python cs336_basics/pretokenization.py`)