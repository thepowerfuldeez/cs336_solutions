use once_cell::sync::Lazy;
use pyo3::{prelude::*, types::PyModule, types::PyBytes, Bound};
use regex::{Regex, escape};
use fancy_regex::Regex as FRegex;
use ahash::AHashMap as Map;
use std::collections::HashMap;
use simdutf::validate_utf8;

// Token pattern (look-around removed for speed).
static TOKEN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
        .expect("valid regex")
});


// use pcre2::bytes::Regex as PcreRe;
// static SLOW_RE: Lazy<PcreRe> = Lazy::new(|| {
//     let mut b = pcre2::bytes::RegexBuilder::new();
//     b.utf(true).ucp(true);
//     if pcre2::is_jit_available() {
//         b.jit(true);
//     }
//     // If this fails because of JIT, rebuild without it.
//     match b.build(TOKEN_RE_SLOW) {
//         Ok(r) => r,
//         Err(_) => {
//             pcre2::bytes::RegexBuilder::new()
//                 .utf(true)
//                 .ucp(true)
//                 .build(TOKEN_RE_SLOW)
//                 .unwrap()
//         }
//     }
// });

static SLOW_RE: Lazy<FRegex> = Lazy::new(|| {
    FRegex::new(r#"\'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#).unwrap()
});


fn needs_lookahead(s: &str) -> bool {
    // return false;
    // cheap-ish check: if there's any trailing whitespace or two consecutive whitespace
    // chars somewhere, the (?!\S) branch can behave differently.
    let mut prev_w = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if prev_w {
                return true; // whitespace followed by whitespace → lookahead can fire
            }
            prev_w = true;
        } else {
            prev_w = false;
        }
    }
    // also if final char(s) are whitespace
    s.chars().last().map_or(false, |c| c.is_whitespace())
}


/// Ignore invalid UTF-8 bytes (drop them rather than replacing with U+FFFD).
fn decode_utf8_ignore(bytes: &[u8]) -> String {
    if validate_utf8(&bytes) {
        // Safe because we just validated it's valid UTF-8.
        unsafe { String::from_utf8_unchecked(bytes.to_vec()) }
    } else {
        // “Ignore errors”: replace invalid sequences with U+FFFD (�).
        String::from_utf8_lossy(&bytes).to_string()
    }
    
    // let mut out = String::with_capacity(bytes.len());
    // let mut i = 0;
    // while i < bytes.len() {
    //     match std::str::from_utf8(&bytes[i..]) {
    //         Ok(valid) => {
    //             out.push_str(valid);
    //             break;
    //         }
    //         Err(e) => {
    //             let good = e.valid_up_to();
    //             if good > 0 {
    //                 // Safety: just validated
    //                 out.push_str(unsafe {
    //                     std::str::from_utf8_unchecked(&bytes[i .. i + good])
    //                 });
    //             }
    //             // Skip the single invalid byte
    //             i += good + 1;
    //         }
    //     }
    // }
    // out
}

#[pyclass]
pub struct Splitter {
    split_re: Regex,
}

#[pymethods]
impl Splitter {
    #[new]
    fn new(special_token: String) -> PyResult<Self> {
        let split_re = Regex::new(&escape(&special_token)).unwrap();
        Ok(Self { split_re })
    }

    /// Split a bytes or str object into token-like substrings.
    /// Pass `bytes` for best performance.
    #[pyo3(signature = (chunk))]
    fn split(&self, chunk: Bound<'_, PyAny>) -> PyResult<HashMap<String, usize>> {
        let text: String = if let Ok(b) = chunk.downcast::<PyBytes>() {
            decode_utf8_ignore(b.as_bytes())
        } else if let Ok(s) = chunk.extract::<&str>() {
            s.to_owned()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Expected bytes or str"));
        };

        self.split_counts_internal(&text)
    }

    /// Read a byte range [start, end) from a file and split.
    #[pyo3(signature = (filepath, start, end))]
    fn seek_and_split<'py>(&self, filepath: &str, start: u64, end: u64) -> PyResult<HashMap<String, usize>> {
        if end < start {
            return Err(pyo3::exceptions::PyValueError::new_err("end < start"));
        }
        use std::fs::File;
        use std::io::{Read, Seek, SeekFrom};

        let mut file = File::open(filepath)?;
        file.seek(SeekFrom::Start(start))?;
        let mut buf = vec![0u8; (end - start) as usize];
        file.read_exact(&mut buf)?;
        let text = decode_utf8_ignore(&buf);

        self.split_counts_internal(&text)
        // let tokens = self.split_counts_internal(&text)?;

        // let mut counts: HashMap<String, usize> = HashMap::new();
        // for t in tokens { *counts.entry(t).or_insert(0) += 1; }
        // Ok(counts)
    }
}

impl Splitter {
    // fn split_internal(&self, text: &str) -> PyResult<Vec<String>> {
    //     let mut tokens: Vec<String> = Vec::new();
    //     for seg in self.split_re.split(text) {
    //         collect_tokens(&seg, &mut tokens);
    //     }
    //     Ok(tokens)
    // }
    #[inline]
    fn split_counts_internal(&self, text: &str) -> PyResult<HashMap<String, usize>> {
        let mut counts: Map<String, usize> = Map::default();
        // Reserve a bit to avoid rehashing — tune this to your data.
        counts.reserve(text.len() / 8);

        for seg in self.split_re.split(text) {
            collect_counts(seg, &mut counts);
        }
        Ok(counts.into_iter().collect())
    }
}

// fn collect_tokens(segment: &str, out: &mut Vec<String>) {
//     if segment.is_empty() {
//         return;
//     }

//     if needs_lookahead(segment) {
//         // exact but slower path
//         for m in SLOW_RE.find_iter(segment) {
//             if let Ok(mat) = m {
//                 out.push(mat.as_str().to_owned());
//             }
//         }
//     } else {
//         // fast path (no lookahead needed)
//         out.extend(TOKEN_RE.find_iter(segment).map(|m| m.as_str().to_owned()));
//     }
// }

#[inline]
fn collect_counts(segment: &str, counts: &mut Map<String, usize>) {
    if segment.is_empty() {
        return;
    }

    // Very cheap heuristic: if the last char is whitespace, the lookahead branch may trigger.
    // If that's still too often, just always take the slow path for correctness.
    let needs_slow = needs_lookahead(segment);

    if needs_slow {
        // ----- PCRE2 bytes path -----
        // let bytes = segment.as_bytes();
        // let mut start = 0;
        // while let Ok(Some(m)) = SLOW_RE.find_at(bytes, start) {
        //     let s = unsafe { std::str::from_utf8_unchecked(&bytes[m.start()..m.end()]) };
        //     bump(counts, s);
        //     start = m.end();
        // }

        // ----- If using fancy-regex instead -----
        for m in SLOW_RE.find_iter(segment) {
            bump(counts, m.unwrap().as_str());
        }
    } else {
        for m in TOKEN_RE.find_iter(segment) {
            bump(counts, m.as_str());
        }
    }
}

#[inline]
fn bump(map: &mut Map<String, usize>, tok: &str) {
    if let Some(v) = map.get_mut(tok) {
        *v += 1;
    } else {
        map.insert(tok.to_owned(), 1);
    }
}

#[pymodule]
fn fastsplit(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Splitter>()?;
    Ok(())
}