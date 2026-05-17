//! Loader for `bit_counts_ECFP_fp_size<N>.csv` files produced by the
//! `molecular-fingerprint-bucket-counts` pipeline.
//!
//! The file format is:
//!
//! ```text
//! # total_molecules: <N>
//! bit_position,count,fraction
//! 0,221048,0.00179703
//! 1,292412,0.00237719
//! ...
//! ```
//!
//! The `fraction` column gives the marginal probability `p_i` that each bin
//! is set across the source corpus. We use it directly as the per-bin
//! frequency for the BCE class reweighting.

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use crate::{AppResult, invalid_input};

/// Loads the `fraction` column from a bit-counts CSV at `path`, validating
/// that exactly `expected_width` rows are present and that every fraction
/// lies in `[0, 1]`. Returns the frequencies in `bit_position` order.
///
/// Lines starting with `#` are treated as comments. The first non-comment
/// line is expected to be the header `bit_position,count,fraction`.
pub fn load_bit_frequencies(path: &Path, expected_width: usize) -> AppResult<Vec<f32>> {
    let file = File::open(path).map_err(|err| {
        invalid_input(format!(
            "failed to open bit-counts file {}: {err}",
            path.display()
        ))
    })?;
    let mut reader = BufReader::new(file).lines();

    // Skip comment lines.
    let header = loop {
        let Some(line) = reader.next() else {
            return Err(invalid_input(format!(
                "bit-counts file {} is empty",
                path.display()
            )));
        };
        let line = line.map_err(|err| {
            invalid_input(format!(
                "failed to read bit-counts file {}: {err}",
                path.display()
            ))
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        break line;
    };
    let header_fields: Vec<&str> = header.split(',').map(str::trim).collect();
    if header_fields != ["bit_position", "count", "fraction"] {
        return Err(invalid_input(format!(
            "unexpected bit-counts header {header:?}; expected bit_position,count,fraction"
        )));
    }

    let mut frequencies = vec![0.0_f32; expected_width];
    let mut seen = vec![false; expected_width];
    let mut count_rows = 0_usize;
    for (row_index, line) in reader.enumerate() {
        let line = line.map_err(|err| {
            invalid_input(format!(
                "failed to read bit-counts file {}: {err} at row {row_index}",
                path.display()
            ))
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.split(',');
        let bit_position: usize = parts
            .next()
            .ok_or_else(|| invalid_input(format!("missing bit_position at row {row_index}")))?
            .trim()
            .parse()
            .map_err(|err| {
                invalid_input(format!("invalid bit_position at row {row_index}: {err}"))
            })?;
        let _count = parts
            .next()
            .ok_or_else(|| invalid_input(format!("missing count at row {row_index}")))?;
        let fraction: f32 = parts
            .next()
            .ok_or_else(|| invalid_input(format!("missing fraction at row {row_index}")))?
            .trim()
            .parse()
            .map_err(|err| invalid_input(format!("invalid fraction at row {row_index}: {err}")))?;
        if bit_position >= expected_width {
            return Err(invalid_input(format!(
                "bit_position {bit_position} out of range for fingerprint width {expected_width}"
            )));
        }
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(invalid_input(format!(
                "fraction at bit_position {bit_position} must be in [0, 1], got {fraction}"
            )));
        }
        if seen[bit_position] {
            return Err(invalid_input(format!(
                "duplicate bit_position {bit_position} in {}",
                path.display()
            )));
        }
        frequencies[bit_position] = fraction;
        seen[bit_position] = true;
        count_rows += 1;
    }
    if count_rows != expected_width {
        return Err(invalid_input(format!(
            "bit-counts file {} has {count_rows} rows; expected {expected_width}",
            path.display()
        )));
    }
    Ok(frequencies)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn write_fixture(text: &str) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().expect("tempfile");
        file.as_file_mut()
            .write_all(text.as_bytes())
            .expect("write");
        file
    }

    #[test]
    fn parses_three_rows_in_position_order() {
        let csv = "# total_molecules: 100\n\
                   bit_position,count,fraction\n\
                   0,10,0.10\n\
                   2,30,0.30\n\
                   1,20,0.20\n";
        let file = write_fixture(csv);
        let frequencies = load_bit_frequencies(file.path(), 3).expect("parse");
        assert_eq!(frequencies, vec![0.10_f32, 0.20, 0.30]);
    }

    #[test]
    fn rejects_mismatched_width() {
        let csv = "bit_position,count,fraction\n0,1,0.1\n1,2,0.2\n";
        let file = write_fixture(csv);
        assert!(load_bit_frequencies(file.path(), 3).is_err());
    }

    #[test]
    fn rejects_out_of_range_fraction() {
        let csv = "bit_position,count,fraction\n0,1,1.5\n";
        let file = write_fixture(csv);
        assert!(load_bit_frequencies(file.path(), 1).is_err());
    }
}
