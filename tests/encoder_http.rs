//! Verifies `MoleculeEncoder::from_url` against a local HTTP fixture server.
//!
//! Spins up a one-shot TCP listener that serves the model artifacts once,
//! then proves the cache short-circuit by calling `from_url` again with
//! the server gone — the bytes are read straight from the tempdir cache.
//! Uses [`MoleculeEncoder::from_url_with_cache_root`] so the test never
//! touches the real `~/.cache/` directory.

#![cfg(all(feature = "ndarray", feature = "http"))]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::thread;

use burn::backend::ndarray::NdArrayDevice;
use burn::module::Module;
use burn::record::DefaultRecorder;
use molecular_autoencoder::{MoleculeAutoencoderConfig, MoleculeEncoder};

type B = burn::backend::NdArray<f32, i64>;

const FINGERPRINT_SIZE: usize = 4096;
const LATENT_WIDTH: usize = 8;

fn save_tiny_checkpoint(dir: &Path) {
    let config = MoleculeAutoencoderConfig::symmetric(FINGERPRINT_SIZE, LATENT_WIDTH, vec![16]);
    config
        .save_json(dir.join("model-config.json"))
        .expect("save config");
    let device = NdArrayDevice::default();
    let model = config.init::<B>(&device);
    model
        .save_file(dir.join("model"), &DefaultRecorder::default())
        .expect("save model");
}

fn serve_one_get(listener: &TcpListener, body: &[u8]) {
    let (mut stream, _) = listener.accept().expect("accept");
    drain_request(&mut stream);
    let header = format!(
        "HTTP/1.1 200 OK\r\n\
         Content-Length: {}\r\n\
         Content-Type: application/octet-stream\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes()).expect("write header");
    stream.write_all(body).expect("write body");
    stream.flush().ok();
}

fn drain_request(stream: &mut TcpStream) {
    // Reqwest sends a small GET request; one read of the buffer clears
    // enough bytes for the server to respond cleanly.
    let mut buf = [0u8; 4096];
    let _ = stream.read(&mut buf);
}

#[tokio::test(flavor = "current_thread")]
async fn from_url_fetches_artifacts_and_caches_them_on_disk() {
    let dir = tempfile::tempdir().expect("tempdir");
    save_tiny_checkpoint(dir.path());
    let config_bytes = std::fs::read(dir.path().join("model-config.json")).expect("config bytes");
    let record_bytes = std::fs::read(dir.path().join("model.mpk")).expect("record bytes");

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let server = thread::spawn(move || {
        serve_one_get(&listener, &config_bytes);
        serve_one_get(&listener, &record_bytes);
    });

    let cache_root = tempfile::tempdir().expect("cache tempdir");
    let cache_path = Some(cache_root.path().to_path_buf());

    let config_url = format!("http://127.0.0.1:{port}/model-config.json");
    let record_url = format!("http://127.0.0.1:{port}/model.mpk");

    let mut encoder = MoleculeEncoder::<B>::from_url_with_cache_root(
        &config_url,
        &record_url,
        NdArrayDevice::default(),
        cache_path.clone(),
    )
    .await
    .expect("from_url first call");
    server.join().expect("server thread join");

    let smiles: Vec<String> = vec!["CCO".to_string(), "c1ccccc1".to_string()];
    let rows = encoder.encode(&smiles).expect("encode");
    assert_eq!(rows.len(), 2);
    for slot in &rows {
        let row = slot.as_ref().expect("valid smiles");
        assert_eq!(row.latent.len(), LATENT_WIDTH);
        assert!(row.latent.iter().all(|v| v.is_finite()));
    }

    // The on-disk cache must now contain both artifacts under one scope.
    let scopes: Vec<_> = std::fs::read_dir(cache_root.path())
        .expect("read cache root")
        .filter_map(Result::ok)
        .collect();
    assert_eq!(scopes.len(), 1, "expected one cache scope directory");
    let scope_path = scopes[0].path();
    assert!(scope_path.join("model-config.json").exists());
    assert!(scope_path.join("model.mpk").exists());

    // Second call: no server is listening. The cache must answer entirely
    // from disk, so this MUST succeed.
    let mut second = MoleculeEncoder::<B>::from_url_with_cache_root(
        &config_url,
        &record_url,
        NdArrayDevice::default(),
        cache_path,
    )
    .await
    .expect("from_url second call (cache hit)");
    let again = second.encode(&smiles).expect("encode #2");
    for (left, right) in rows.iter().zip(again.iter()) {
        let left_row = left.as_ref().expect("row #1");
        let right_row = right.as_ref().expect("row #2");
        assert_eq!(left_row.latent, right_row.latent);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn from_url_propagates_http_error_status() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        drain_request(&mut stream);
        let response = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        stream.write_all(response).ok();
    });

    let cache_root = tempfile::tempdir().expect("cache tempdir");
    let result = MoleculeEncoder::<B>::from_url_with_cache_root(
        &format!("http://127.0.0.1:{port}/missing-config.json"),
        &format!("http://127.0.0.1:{port}/missing-model.mpk"),
        NdArrayDevice::default(),
        Some(cache_root.path().to_path_buf()),
    )
    .await;
    server.join().ok();

    match result {
        Err(molecular_autoencoder::Error::Network { url, message }) => {
            assert!(url.contains("missing-config.json"));
            assert!(message.contains("404"), "got: {message}");
        }
        Err(other) => panic!("unexpected error variant: {other:?}"),
        Ok(_) => panic!("non-success status must surface as Error::Network"),
    }
}
