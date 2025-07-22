use once_cell::sync::Lazy;
use pyo3::{prelude::*, types::PyModule, types::PyBytes, Bound};
use regex::Regex;
use std::collections::HashMap;

// Token pattern (look-around removed for speed).
static TOKEN_RE: Lazy<Regex> = Lazy::new(|| {
    // Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
    Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+\z|\s+")
        .expect("valid regex")
});

/// Ignore invalid UTF-8 bytes (drop them rather than replacing with U+FFFD).
fn decode_utf8_ignore(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match std::str::from_utf8(&bytes[i..]) {
            Ok(valid) => {
                out.push_str(valid);
                break;
            }
            Err(e) => {
                let good = e.valid_up_to();
                if good > 0 {
                    // Safety: just validated
                    out.push_str(unsafe {
                        std::str::from_utf8_unchecked(&bytes[i .. i + good])
                    });
                }
                // Skip the single invalid byte
                i += good + 1;
            }
        }
    }
    out
}

#[pyclass]
pub struct Splitter {
    special_token: String,
}

#[pymethods]
impl Splitter {
    #[new]
    fn new(special_token: String) -> PyResult<Self> {
        let special_token = special_token;
        Ok(Self { special_token })
    }

    /// Split a bytes or str object into token-like substrings.
    /// Pass `bytes` for best performance.
    #[pyo3(signature = (chunk))]
    fn split(&self, chunk: Bound<'_, PyAny>) -> PyResult<Vec<String>> {
        let text: String = if let Ok(b) = chunk.downcast::<PyBytes>() {
            decode_utf8_ignore(b.as_bytes())
        } else if let Ok(s) = chunk.extract::<&str>() {
            s.to_owned()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Expected bytes or str"));
        };

        self.split_internal(&text)
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

        let tokens = self.split_internal(&text)?;

        let mut counts: HashMap<String, usize> = HashMap::new();
        for t in tokens { *counts.entry(t).or_insert(0) += 1; }
        Ok(counts)
    }
}

impl Splitter {
    fn split_internal(&self, text: &str) -> PyResult<Vec<String>> {
        let mut tokens: Vec<String> = Vec::new();
        for seg in text.split(self.special_token.as_str()) {
            collect_tokens(seg, &mut tokens);
        }
        Ok(tokens)
    }
}

fn collect_tokens(segment: &str, out: &mut Vec<String>) {
    if segment.is_empty() {
        return;
    }
    for m in TOKEN_RE.find_iter(segment) {
        out.push(m.as_str().to_owned());
    }
}

#[pymodule]
fn fastsplit(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Splitter>()?;
    Ok(())
}