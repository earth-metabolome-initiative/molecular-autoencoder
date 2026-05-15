//! Command-line argument parsing for the `train` bin.

use std::{env, path::PathBuf};

use molecular_autoencoder::MoleculeAutoencoderConfig;

use crate::{AppResult, invalid_input};

/// Default Rayon workers for dataset preprocessing.
pub const DEFAULT_PREPROCESS_THREADS: usize = 64;
/// Default rows written per cached shard.
pub const DEFAULT_ROWS_PER_SHARD: usize = 10_000_000;
/// Default number of host-side dataloader worker threads.
pub const DEFAULT_LOADER_WORKERS: usize = 2;
/// Default device-side prefetch depth.
pub const DEFAULT_DEVICE_PREFETCH_BATCHES: usize = 0;
/// Default sampling stride for expensive training metrics.
pub const DEFAULT_METRIC_EVERY: usize = 10;
/// Smallest valid ZINC20 chunk index.
pub const ZINC20_FIRST_CHUNK: u8 = 1;
/// Largest valid ZINC20 chunk index.
pub const ZINC20_LAST_CHUNK: u8 = 20;

/// Datasets selected for ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceSelection {
    /// Only PubChem records.
    PubChem,
    /// Only ZINC20 records over an inclusive chunk range.
    Zinc20 { first_chunk: u8, last_chunk: u8 },
    /// Both PubChem and a ZINC20 chunk range.
    PubChemAndZinc20 { first_chunk: u8, last_chunk: u8 },
}

impl SourceSelection {
    /// Overrides the ZINC20 chunk range if this selection ingests ZINC20.
    pub fn with_zinc20_chunks(self, first_chunk: u8, last_chunk: u8) -> AppResult<Self> {
        match self {
            Self::PubChem => Err(invalid_input(
                "--zinc20-chunks is only valid when --datasets includes `zinc20`",
            )),
            Self::Zinc20 { .. } => Ok(Self::Zinc20 {
                first_chunk,
                last_chunk,
            }),
            Self::PubChemAndZinc20 { .. } => Ok(Self::PubChemAndZinc20 {
                first_chunk,
                last_chunk,
            }),
        }
    }

    /// Returns the manifest source label that identifies this selection.
    pub fn manifest_source(self) -> String {
        match self {
            Self::PubChem => "pubchem-smiles".to_string(),
            Self::Zinc20 {
                first_chunk,
                last_chunk,
            } => format!("zinc20-smiles chunks {first_chunk}..={last_chunk}"),
            Self::PubChemAndZinc20 {
                first_chunk,
                last_chunk,
            } => format!("pubchem-smiles + zinc20-smiles chunks {first_chunk}..={last_chunk}"),
        }
    }
}

/// Fully-parsed training CLI configuration.
#[derive(Debug, Clone)]
pub struct Args {
    pub manifest_path: PathBuf,
    pub cache_dir: Option<PathBuf>,
    pub source_selection: Option<SourceSelection>,
    pub checkpoint_dir: PathBuf,
    pub rows_per_shard: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub latent_width: usize,
    pub latent_noise_std: f64,
    pub hidden_widths: Vec<usize>,
    pub checkpoint_every: usize,
    pub validate_every: usize,
    pub max_train_batches: Option<usize>,
    pub max_valid_batches: Option<usize>,
    pub loader_workers: usize,
    pub device_prefetch_batches: usize,
    pub loader_profile_every: usize,
    pub metric_every: usize,
    pub descriptor_weight: f64,
    pub tanimoto_ranking_weight: f64,
    pub tanimoto_ranking_latent_temperature: f64,
    pub tanimoto_ranking_metric_temperature: f64,
    pub tanimoto_ranking_min_gap: f64,
    pub tanimoto_ranking_candidates: usize,
    pub tanimoto_ranking_pairs_per_batch: usize,
    pub resume: bool,
    pub force_preprocess: bool,
    pub preprocess_threads: usize,
    pub cuda_device: usize,
}

impl Args {
    /// Parses the process command line.
    pub fn parse() -> AppResult<Self> {
        let raw: Vec<String> = env::args().skip(1).collect();
        if raw.iter().any(|arg| arg == "--help" || arg == "-h") {
            return Err(invalid_input(Self::usage()));
        }
        let mut values = raw.into_iter();
        let shards_dir = next_value(&mut values, "<shards-dir-or-manifest>")?;
        let checkpoint_dir = PathBuf::from(next_value(&mut values, "<checkpoint-dir>")?);

        let shards_path = PathBuf::from(&shards_dir);
        let (manifest_path, cache_dir) = if shards_path.is_dir() || !shards_path.exists() {
            (shards_path.join("manifest.json"), Some(shards_path))
        } else {
            (shards_path, None)
        };

        let mut args = Self {
            manifest_path,
            cache_dir,
            source_selection: None,
            checkpoint_dir,
            rows_per_shard: DEFAULT_ROWS_PER_SHARD,
            epochs: 10,
            batch_size: 4096,
            learning_rate: 1.0e-3,
            latent_width: 512,
            latent_noise_std: 0.02,
            hidden_widths: vec![4096, 2048, 1024],
            checkpoint_every: 1,
            validate_every: 1,
            max_train_batches: None,
            max_valid_batches: Some(64),
            loader_workers: DEFAULT_LOADER_WORKERS,
            device_prefetch_batches: DEFAULT_DEVICE_PREFETCH_BATCHES,
            loader_profile_every: 0,
            metric_every: DEFAULT_METRIC_EVERY,
            descriptor_weight: 0.05,
            tanimoto_ranking_weight: 0.10,
            tanimoto_ranking_latent_temperature: 0.10,
            tanimoto_ranking_metric_temperature: 0.10,
            tanimoto_ranking_min_gap: 0.05,
            tanimoto_ranking_candidates: 4,
            tanimoto_ranking_pairs_per_batch: 0,
            resume: false,
            force_preprocess: false,
            preprocess_threads: DEFAULT_PREPROCESS_THREADS,
            cuda_device: 0,
        };

        let mut zinc20_chunks: Option<(u8, u8)> = None;

        while let Some(flag) = values.next() {
            match flag.as_str() {
                "--datasets" => {
                    let value = next_value(&mut values, &flag)?;
                    args.source_selection = Some(parse_datasets_csv(&value)?);
                }
                "--epochs" => args.epochs = parse_value(&mut values, &flag)?,
                "--batch-size" => args.batch_size = parse_value(&mut values, &flag)?,
                "--learning-rate" => args.learning_rate = parse_value(&mut values, &flag)?,
                "--latent-width" => args.latent_width = parse_value(&mut values, &flag)?,
                "--latent-noise-std" => {
                    args.latent_noise_std = parse_value(&mut values, &flag)?;
                }
                "--hidden-widths" => {
                    let value = next_value(&mut values, &flag)?;
                    args.hidden_widths = parse_hidden_widths(&value)?;
                }
                "--rows-per-shard" => args.rows_per_shard = parse_value(&mut values, &flag)?,
                "--zinc20-chunks" => {
                    let value = next_value(&mut values, &flag)?;
                    zinc20_chunks = Some(parse_zinc20_chunks(&value)?);
                }
                "--checkpoint-every" => args.checkpoint_every = parse_value(&mut values, &flag)?,
                "--validate-every" => args.validate_every = parse_value(&mut values, &flag)?,
                "--max-train-batches" => {
                    args.max_train_batches = Some(parse_value(&mut values, &flag)?);
                }
                "--max-valid-batches" => {
                    args.max_valid_batches = Some(parse_value(&mut values, &flag)?);
                }
                "--loader-workers" => args.loader_workers = parse_value(&mut values, &flag)?,
                "--device-prefetch-batches" => {
                    args.device_prefetch_batches = parse_value(&mut values, &flag)?;
                }
                "--loader-profile-every" => {
                    args.loader_profile_every = parse_value(&mut values, &flag)?;
                }
                "--metric-every" => args.metric_every = parse_positive_value(&mut values, &flag)?,
                "--descriptor-weight" => {
                    args.descriptor_weight = parse_value(&mut values, &flag)?;
                }
                "--tanimoto-ranking-weight" => {
                    args.tanimoto_ranking_weight = parse_value(&mut values, &flag)?;
                }
                "--tanimoto-ranking-margin" | "--tanimoto-ranking-latent-temperature" => {
                    args.tanimoto_ranking_latent_temperature = parse_value(&mut values, &flag)?;
                }
                "--tanimoto-ranking-metric-temperature" => {
                    args.tanimoto_ranking_metric_temperature = parse_value(&mut values, &flag)?;
                }
                "--tanimoto-ranking-min-gap" => {
                    args.tanimoto_ranking_min_gap = parse_value(&mut values, &flag)?;
                }
                "--tanimoto-ranking-candidates" => {
                    args.tanimoto_ranking_candidates = parse_positive_value(&mut values, &flag)?;
                }
                "--tanimoto-ranking-pairs-per-batch" => {
                    args.tanimoto_ranking_pairs_per_batch = parse_value(&mut values, &flag)?;
                }
                "--full-validation" => args.max_valid_batches = None,
                "--resume" => args.resume = true,
                "--force-preprocess" => args.force_preprocess = true,
                "--preprocess-threads" => {
                    args.preprocess_threads = parse_positive_value(&mut values, &flag)?;
                }
                "--cuda-device" => args.cuda_device = parse_value(&mut values, &flag)?,
                "--help" | "-h" => return Err(invalid_input(Self::usage())),
                unknown => {
                    return Err(invalid_input(format!(
                        "unknown argument `{unknown}`\n{}",
                        Self::usage()
                    )));
                }
            }
        }

        if let Some((first_chunk, last_chunk)) = zinc20_chunks {
            let Some(selection) = args.source_selection else {
                return Err(invalid_input(
                    "--zinc20-chunks requires --datasets to include `zinc20`",
                ));
            };
            args.source_selection = Some(selection.with_zinc20_chunks(first_chunk, last_chunk)?);
        }

        Ok(args)
    }

    /// Returns the usage string used by `--help` and on error.
    pub fn usage() -> String {
        "usage: train <shards-dir|manifest.json> <checkpoint-dir> \
         [--datasets pubchem,zinc20] [--zinc20-chunks FIRST-LAST] \
         [--epochs N] [--batch-size N] [--learning-rate LR] [--latent-width N] \
         [--latent-noise-std STD] \
         [--hidden-widths 4096,2048,1024] [--checkpoint-every N] \
         [--validate-every N] [--max-train-batches N] [--max-valid-batches N] \
         [--loader-workers N] [--device-prefetch-batches N] \
         [--metric-every N] [--loader-profile-every N] \
         [--descriptor-weight W] \
         [--tanimoto-ranking-weight W] \
         [--tanimoto-ranking-latent-temperature T] \
         [--tanimoto-ranking-metric-temperature T] \
         [--tanimoto-ranking-min-gap G] [--tanimoto-ranking-candidates N] \
         [--tanimoto-ranking-pairs-per-batch N] \
         [--full-validation] \
         [--rows-per-shard N] [--preprocess-threads N] [--force-preprocess] \
         [--resume] [--cuda-device N]"
            .to_string()
    }

    /// Builds a [`MoleculeAutoencoderConfig`] from the CLI components.
    pub fn to_model_config(&self, fingerprint_size: usize) -> MoleculeAutoencoderConfig {
        MoleculeAutoencoderConfig::from_components(
            fingerprint_size,
            self.latent_width,
            self.hidden_widths.clone(),
            self.descriptor_weight,
            self.tanimoto_ranking_weight,
            self.latent_noise_std,
            self.tanimoto_ranking_latent_temperature,
            self.tanimoto_ranking_metric_temperature,
            self.tanimoto_ranking_min_gap,
            self.tanimoto_ranking_candidates,
            self.tanimoto_ranking_pairs_per_batch,
        )
    }
}

fn parse_datasets_csv(value: &str) -> AppResult<SourceSelection> {
    let mut want_pubchem = false;
    let mut want_zinc20 = false;
    for raw in value.split(',') {
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        match token.to_ascii_lowercase().as_str() {
            "pubchem" | "pubchem-smiles" => {
                if want_pubchem {
                    return Err(invalid_input("--datasets lists `pubchem` more than once"));
                }
                want_pubchem = true;
            }
            "zinc20" | "zinc20-smiles" => {
                if want_zinc20 {
                    return Err(invalid_input("--datasets lists `zinc20` more than once"));
                }
                want_zinc20 = true;
            }
            other => {
                return Err(invalid_input(format!(
                    "unknown dataset `{other}` in --datasets; valid values are `pubchem` and `zinc20`"
                )));
            }
        }
    }

    match (want_pubchem, want_zinc20) {
        (true, true) => Ok(SourceSelection::PubChemAndZinc20 {
            first_chunk: ZINC20_FIRST_CHUNK,
            last_chunk: ZINC20_LAST_CHUNK,
        }),
        (true, false) => Ok(SourceSelection::PubChem),
        (false, true) => Ok(SourceSelection::Zinc20 {
            first_chunk: ZINC20_FIRST_CHUNK,
            last_chunk: ZINC20_LAST_CHUNK,
        }),
        (false, false) => Err(invalid_input(
            "--datasets must list at least one of `pubchem` or `zinc20`",
        )),
    }
}

fn next_value(values: &mut impl Iterator<Item = String>, flag: &str) -> AppResult<String> {
    values
        .next()
        .ok_or_else(|| invalid_input(format!("missing value for `{flag}`")))
}

fn parse_value<T>(values: &mut impl Iterator<Item = String>, flag: &str) -> AppResult<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = next_value(values, flag)?;
    value
        .parse::<T>()
        .map_err(|source| invalid_input(format!("invalid value for `{flag}`: {source}")))
}

fn parse_positive_value(values: &mut impl Iterator<Item = String>, flag: &str) -> AppResult<usize> {
    let value = parse_value::<usize>(values, flag)?;
    if value == 0 {
        return Err(invalid_input(format!("`{flag}` must be greater than zero")));
    }
    Ok(value)
}

fn parse_zinc20_chunks(value: &str) -> AppResult<(u8, u8)> {
    let value = value.trim();
    let (first, last) = if let Some((first, last)) = value.split_once("..=") {
        (first, last)
    } else if let Some((first, last)) = value.split_once('-') {
        (first, last)
    } else {
        (value, value)
    };
    let first_chunk = first.parse::<u8>().map_err(|source| {
        invalid_input(format!("invalid ZINC20 first chunk `{first}`: {source}"))
    })?;
    let last_chunk = last
        .parse::<u8>()
        .map_err(|source| invalid_input(format!("invalid ZINC20 last chunk `{last}`: {source}")))?;
    if first_chunk < ZINC20_FIRST_CHUNK
        || last_chunk > ZINC20_LAST_CHUNK
        || first_chunk > last_chunk
    {
        return Err(invalid_input(format!(
            "ZINC20 chunks must be an inclusive range within {ZINC20_FIRST_CHUNK}..={ZINC20_LAST_CHUNK}"
        )));
    }
    Ok((first_chunk, last_chunk))
}

fn parse_hidden_widths(value: &str) -> AppResult<Vec<usize>> {
    let widths = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .map_err(|source| invalid_input(format!("invalid hidden width `{part}`: {source}")))
        })
        .collect::<AppResult<Vec<_>>>()?;

    if widths.is_empty() {
        Err(invalid_input("at least one hidden width is required"))
    } else {
        Ok(widths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datasets_csv_accepts_both_orderings() {
        assert_eq!(
            parse_datasets_csv("pubchem").expect("pubchem only"),
            SourceSelection::PubChem
        );
        assert_eq!(
            parse_datasets_csv("zinc20").expect("zinc20 only"),
            SourceSelection::Zinc20 {
                first_chunk: ZINC20_FIRST_CHUNK,
                last_chunk: ZINC20_LAST_CHUNK,
            }
        );
        let both = parse_datasets_csv("zinc20, pubchem").expect("both datasets");
        assert_eq!(
            both,
            SourceSelection::PubChemAndZinc20 {
                first_chunk: ZINC20_FIRST_CHUNK,
                last_chunk: ZINC20_LAST_CHUNK,
            }
        );
    }

    #[test]
    fn datasets_csv_rejects_duplicates_and_unknown() {
        assert!(parse_datasets_csv("pubchem,pubchem").is_err());
        assert!(parse_datasets_csv("zinc20,zinc20").is_err());
        assert!(parse_datasets_csv("chembl").is_err());
        assert!(parse_datasets_csv("").is_err());
    }

    #[test]
    fn zinc20_chunks_apply_only_when_zinc20_is_selected() {
        let pubchem_only = SourceSelection::PubChem;
        assert!(pubchem_only.with_zinc20_chunks(2, 4).is_err());

        let zinc20 = SourceSelection::Zinc20 {
            first_chunk: 1,
            last_chunk: 20,
        };
        let updated = zinc20
            .with_zinc20_chunks(2, 5)
            .expect("zinc20 selection accepts chunks");
        assert_eq!(
            updated,
            SourceSelection::Zinc20 {
                first_chunk: 2,
                last_chunk: 5
            }
        );
    }
}
