//! den-embed — bge-m3 int8 embedding service (Rust).
//!
//! A 1:1 rewrite of the Python service (server.py) with one goal beyond speed:
//! a tiny idle footprint. The Python process, even with the model unloaded, held
//! ~100 MB (interpreter + numpy + fastapi + onnxruntime arenas that gc can't
//! return). This drops the interpreter entirely; with the model AND tokenizer
//! idle-unloaded it falls back to tens of MB (measured figures in CLAUDE.md),
//! reloading on the next request.
//!
//! Parity is the contract. Corpus vectors and query vectors are only comparable
//! because they pass through ONE canonical path: same tokenizer.json (HuggingFace
//! `tokenizers`, the very crate fastembed wraps), same model_int8.onnx on the same
//! ONNX Runtime CPU provider (via `ort`), same CLS pooling, same L2-normalize, same
//! `round(x*127)` clamp. tests/parity_check.py — run by hand against a live
//! instance and golden vectors, not in CI — pins this against the Python service's
//! actual output.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Duration, Instant};

use anyhow::Context;
use axum::extract::{Query, Request, State};
use axum::http::header::{
    ACCESS_CONTROL_ALLOW_HEADERS, ACCESS_CONTROL_ALLOW_METHODS, ACCESS_CONTROL_ALLOW_ORIGIN,
    ACCESS_CONTROL_MAX_AGE, AUTHORIZATION, CACHE_CONTROL, CONTENT_TYPE,
};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use blake2::digest::consts::U16;
use blake2::{Blake2b, Digest};
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

// --- fixed contract (mirror server.py) -------------------------------------
const MODEL_LABEL: &str = "bge-m3";
const DIMS: usize = 1024;

/// Bumped ONLY when this service's vector OUTPUT changes for the same input.
///
/// The crate version cannot serve this purpose, and using it was a bug: den-dataset records the embedder
/// identity with each corpus and refuses to append a different one, so with the version as the identity a
/// release that touched nothing but a log line would invalidate a 37.5k-title corpus and demand hours of
/// re-embedding. Equally, `model` and `dims` alone are too coarse — they say bge-m3/1024 for every
/// generation, including the ORT 1.22 -> 1.28 bump that DID move int8 output.
///
/// So: bump this for an ONNX Runtime upgrade, a model or revision change, a pooling or normalisation
/// change, or anything else that moves the numbers. Do NOT bump it for anything else.
///
/// 1 = the Rust service (ORT 1.28). The Python/ORT-1.22 generation that built the corpus shipping today
/// predates the field entirely and reads back as 0, which is correctly not equal to this.
const VECTOR_EPOCH: u32 = 1;

/// `ort::Error` doesn't implement `std::error::Error`, so `?` can't convert it
/// into `anyhow::Error` directly; map it through its Display.
///
/// Generic over the recovery payload: from ort rc.13 the session-builder methods fail with
/// `Error<SessionBuilder>`, which hands the builder back so a caller could retry, while everything
/// else still fails with `Error<()>`. We do not retry, so both collapse to the same message.
fn ort_err<R>(e: ort::Error<R>) -> anyhow::Error {
    anyhow::anyhow!("ort: {e}")
}

// glibc-only: return free heap arenas to the OS after the model is dropped. Not
// exposed by the `libc` crate, so declare it directly (the image is glibc/debian).
// Guarded for Linux so the service still builds on macOS (dev); a no-op elsewhere.
#[cfg(target_os = "linux")]
extern "C" {
    fn malloc_trim(pad: usize) -> i32;
}
#[cfg(not(target_os = "linux"))]
unsafe fn malloc_trim(_pad: usize) -> i32 {
    0
}

// --- config from env ---------------------------------------------------------
// No name carries an addon prefix: the container is the namespace, so PORT, METRICS_TOKEN and the
// knobs below read the same way in every den addon. There is no host knob: like every other addon it
// binds 0.0.0.0, and staying off the LAN is the container network's job (no published port), not the
// bind address's.
struct Config {
    port: u16,
    onnx_path: String,
    tokenizer_path: String,
    max_chars: usize,
    /// Per-text TOKEN cap. `max_chars` bounds characters, which bounds nothing that matters: at the
    /// 8000-char limit, emoji tokenize to 8003 tokens and Hangul to 8002, so a small request reached
    /// the model's full 8192-token ceiling. Activation memory grows superlinearly with sequence
    /// length — measured peak RSS 1087 MB at 512 tokens, 1219 MB at 1024, 1598 MB at 2048 — against
    /// a 1536 MB cgroup limit, so a ~10 KB POST body was an OOM-kill of the container.
    ///
    /// 512 because this service embeds SEARCH QUERIES; anything near the cap is already not a query.
    max_tokens: usize,
    /// Total TOKENS across one request, batch included — the thing that actually costs time. Per-text
    /// limits bound nothing in aggregate: `max_batch` (512) x `max_chars` (8000) is 4M characters
    /// through a serial loop holding the model lock, measured at ~20-30 minutes of pinned service.
    ///
    /// Counted as `min(chars, max_tokens)` per text, which is an upper bound on what inference will
    /// actually see: a text yields at most one token per character, and truncation caps it at
    /// `max_tokens` regardless. Counting raw characters instead got this backwards in both
    /// directions — it rejected three 8000-char English texts (1536 tokens, about a second of work)
    /// while admitting 32 texts of 512 CJK characters (16384 tokens, ~10 seconds).
    max_request_tokens: usize,
    max_batch: usize,
    max_body_bytes: usize,
    cache_max: usize,
    idle_unload: Option<Duration>,
    intra_threads: usize,
    /// Bearer token for `/metrics`; `None` (unset or blank) turns the route off.
    metrics_token: Option<String>,
    /// One stderr line per request. Read once here, so with it off the only cost is not adding the layer.
    log_requests: bool,
}

/// Read a numeric env var, clamped to `[min, max]`.
///
/// A malformed value is REPORTED, not silently replaced: `IDLE_UNLOAD_SECS='600s'` parsed
/// as nothing and fell back to 0, which means always-warm — ~1 GB resident forever, the exact
/// opposite of what the operator asked for, with no line anywhere saying so. That value lives in
/// /etc/den/env on the box, outside this repo, so nothing reviews it either.
///
/// `max` matters as much as `min`: several of these bound memory, and an out-of-range value silently
/// restores the failure the bound exists to prevent (`MAX_TOKENS=8192` is the OOM again).
fn env_clamped(key: &str, default: usize, min: usize, max: usize) -> usize {
    match std::env::var(key) {
        Err(_) => default,
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(v) => {
                let clamped = v.clamp(min, max);
                if clamped != v {
                    tracing::warn!("{key}={v} is outside {min}..={max}; using {clamped}");
                }
                clamped
            }
            Err(_) => {
                tracing::warn!("{key}={raw:?} is not a number; using the default {default}");
                default
            }
        },
    }
}

impl Config {
    fn from_env() -> Self {
        // Model files are baked into the image (see Dockerfile). Default to the
        // fixed bake paths; overridable for local dev.
        let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "/models".into());
        let onnx_path = std::env::var("ONNX_PATH").unwrap_or_else(|_| format!("{model_dir}/model_int8.onnx"));
        let tokenizer_path =
            std::env::var("TOKENIZER_PATH").unwrap_or_else(|_| format!("{model_dir}/tokenizer.json"));
        // A day is already far past "idle"; anything larger is a typo.
        let idle = env_clamped("IDLE_UNLOAD_SECS", 0, 0, 86_400);
        Self {
            port: std::env::var("PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8080),
            onnx_path,
            tokenizer_path,
            max_chars: env_clamped("MAX_CHARS", 8000, 500, 100_000),
            // 1024 is the ceiling, not the model's: measured peak RSS is 1219 MB at 1024 tokens and
            // 1598 MB at 2048, against a 1536 MB cgroup. Above this the cap stops being a bound.
            max_tokens: env_clamped("MAX_TOKENS", 512, 16, 1024),
            // ~0.33s per 512 tokens measured, so 8192 is ~5s — inside the 10s timeout den-atlas
            // applies to this call. The CEILING has to respect that too, and strictly: 32768 was
            // ~21s (double the timeout the comment cited), and 16384 is ~10.5s, still past it.
            // 12288 is ~7.9s, the largest batch a caller is actually still waiting for.
            max_request_tokens: env_clamped("MAX_REQUEST_TOKENS", 8192, 512, 12_288),
            max_batch: env_clamped("MAX_BATCH", 512, 1, 4096),
            max_body_bytes: env_clamped("MAX_BODY_BYTES", 4 * 1024 * 1024, 64 * 1024, 16 * 1024 * 1024),
            // Each entry is ~4.2 KB, so this is the cache's memory bound too. The ceiling has to
            // hold ALONGSIDE the model, not instead of it: max_tokens at its own ceiling of 1024
            // measures 1219 MB peak, and 65536 entries is ~275 MB — 1494 MB against a 1536 MB
            // cgroup, i.e. both ceilings set at once was an OOM. 32768 is ~137 MB, leaving ~180 MB.
            cache_max: env_clamped("CACHE_MAX_ENTRIES", 8192, 0, 32_768),
            idle_unload: (idle > 0).then(|| Duration::from_secs(idle as u64)),
            // ONNX intra-op threads. Default to all cores (fastembed/ORT default).
            intra_threads: env_clamped("INTRA_THREADS", 0, 0, 256),
            metrics_token: std::env::var("METRICS_TOKEN")
                .ok()
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty()),
            // Unset, empty or "0" is off; anything else is on.
            log_requests: std::env::var("LOG_REQUESTS").is_ok_and(|v| !matches!(v.trim(), "" | "0")),
        }
    }
}

// --- the loaded model (session + tokenizer), held only while warm -----------
struct Model {
    session: Session,
    tokenizer: Tokenizer,
}

impl Model {
    fn load(cfg: &Config) -> anyhow::Result<Self> {
        let started = Instant::now();
        let mut builder = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?;
        if cfg.intra_threads > 0 {
            builder = builder.with_intra_threads(cfg.intra_threads).map_err(ort_err)?;
        }
        let session = builder.commit_from_file(&cfg.onnx_path).map_err(ort_err)?;
        let mut tokenizer = Tokenizer::from_file(&cfg.tokenizer_path).map_err(|e| anyhow::anyhow!("{e}"))?;
        // bge-m3 supports 8192 tokens, but the ceiling that matters here is memory, not the model:
        // see `Config::max_tokens`. The tokenizer.json ships no truncation, so this is where the cap
        // is actually enforced — `max_chars` does not bound tokens for non-Latin text.
        let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: cfg.max_tokens,
            ..Default::default()
        }));
        tracing::info!("loaded model in {}ms", started.elapsed().as_millis());
        Ok(Self { session, tokenizer })
    }
}

/// Marks a failure to load the model, so it is rate-limited apart from a failure of inference itself:
/// a missing model fails every request, and its line must not stand in for a different failure.
#[derive(Debug)]
struct LoadFailed;

impl std::fmt::Display for LoadFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("model load failed")
    }
}

/// The model, loaded first if it is not resident — never loaded yet, or dropped by idle-unload.
fn ensure_loaded<'a>(slot: &'a mut Option<Model>, cfg: &Config) -> anyhow::Result<&'a mut Model> {
    if slot.is_none() {
        *slot = Some(Model::load(cfg).context(LoadFailed)?);
    }
    Ok(slot.as_mut().unwrap())
}

struct AppState {
    cfg: Config,
    model: Mutex<Option<Model>>,
    // millis since `started` of the last embed; drives idle-unload.
    last_used_ms: AtomicU64,
    started: Instant,
    cache: Mutex<Lru>,
    /// When each failure condition was last logged, and how many repeats have gone unlogged since.
    failures: Mutex<HashMap<&'static str, (Option<Instant>, u64)>>,
}

impl AppState {
    fn touch(&self) {
        self.last_used_ms.store(self.started.elapsed().as_millis() as u64, Ordering::Relaxed);
    }
}

// --- tiny LRU: cache is a pure optimization, so its eviction policy does not
// affect output vectors. Counter-recency LRU; evict the least-recent when full.
struct Lru {
    map: HashMap<String, (Vec<i32>, u64)>,
    tick: u64,
    cap: usize,
}

impl Lru {
    fn new(cap: usize) -> Self {
        Self { map: HashMap::new(), tick: 0, cap }
    }
    fn get(&mut self, key: &str) -> Option<Vec<i32>> {
        if self.cap == 0 {
            return None;
        }
        self.tick += 1;
        let tick = self.tick;
        if let Some(entry) = self.map.get_mut(key) {
            entry.1 = tick;
            Some(entry.0.clone())
        } else {
            None
        }
    }
    fn put(&mut self, key: String, val: Vec<i32>) {
        if self.cap == 0 {
            return;
        }
        self.tick += 1;
        self.map.insert(key, (val, self.tick));
        while self.map.len() > self.cap {
            if let Some(oldest) = self.map.iter().min_by_key(|(_, (_, t))| *t).map(|(k, _)| k.clone()) {
                self.map.remove(&oldest);
            } else {
                break;
            }
        }
    }
}

// --- the canonical embedding path ------------------------------------------

/// blake2b-128 hex of `"{MODEL_LABEL}\x00{text}"`, matching server.py's
/// `hashlib.blake2b(..., digest_size=16).hexdigest()`. `text` is already truncated.
fn cache_key(text: &str) -> String {
    let mut h = Blake2b::<U16>::new();
    h.update(MODEL_LABEL.as_bytes());
    h.update(b"\x00");
    h.update(text.as_bytes());
    hex::encode(h.finalize())
}

fn is_blank(text: &str) -> bool {
    text.trim().is_empty()
}

fn zero_vector() -> Vec<i32> {
    vec![0; DIMS]
}

/// L2-normalize in f64 then quantize to int8: `clamp(round_ties_even(x*127), -127, 127)`.
/// numpy's `np.round` is round-half-to-even and `quantize_int8` upcasts to float64
/// before normalizing, so both are matched here.
fn quantize_int8(cls: &[f32]) -> Vec<i32> {
    let mut v: Vec<f64> = cls.iter().map(|&x| x as f64).collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v.iter().map(|&x| ((x * 127.0).round_ties_even() as i64).clamp(-127, 127) as i32).collect()
}

/// Run the model on one already-truncated, non-blank text → CLS-pooled int8 vector.
/// Serialized under the model lock (single-worker, like the Python service), which
/// also lazily loads the model on first use and refreshes the idle timer.
fn infer(state: &AppState, text: &str) -> anyhow::Result<Vec<i32>> {
    let mut guard = state.model.lock().unwrap();
    let model = ensure_loaded(&mut guard, &state.cfg)?;

    let enc = model.tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
    let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&x| x as i64).collect();
    let seq = ids.len();

    let ids = Tensor::from_array(Array2::from_shape_vec((1, seq), ids)?).map_err(ort_err)?;
    let mask = Tensor::from_array(Array2::from_shape_vec((1, seq), mask)?).map_err(ort_err)?;
    let outputs =
        model.session.run(ort::inputs!["input_ids" => ids, "attention_mask" => mask]).map_err(ort_err)?;
    let (_shape, data) = outputs["last_hidden_state"].try_extract_tensor::<f32>().map_err(ort_err)?;
    // last_hidden_state is [1, seq, 1024]; CLS pooling = token 0 = data[0..1024].
    let cls = &data[0..DIMS];
    let out = quantize_int8(cls);

    state.touch();
    Ok(out)
}

fn embed_one(state: &AppState, text: &str) -> anyhow::Result<Vec<i32>> {
    if is_blank(text) {
        return Ok(zero_vector());
    }
    let truncated: String = text.chars().take(state.cfg.max_chars).collect();
    let key = cache_key(&truncated);
    if let Some(hit) = state.cache.lock().unwrap().get(&key) {
        return Ok(hit);
    }
    let vector = infer(state, &truncated)?;
    state.cache.lock().unwrap().put(key, vector.clone());
    Ok(vector)
}

fn embed_many(state: &AppState, texts: &[String]) -> anyhow::Result<Vec<Vec<i32>>> {
    // CLS pooling is padding-invariant, so per-item inference is byte-identical to
    // fastembed's batched path; the content cache handles repeats.
    texts.iter().map(|t| embed_one(state, t)).collect()
}

/// The batch's real token count, capped per text exactly as inference will cap it.
///
/// Counting `min(chars, max_tokens)` instead was UNSOUND, and not marginally: the tokenizer's
/// normalizer is SentencePiece's `Precompiled` NFKC charsmap, which EXPANDS a single scalar before
/// the model sees it. `㌚` (U+331A) yields 5.05 tokens per character, and 1187 codepoints across the
/// CJK-compatibility, Arabic-presentation and halfwidth blocks exceed 1 token/char. A 49 KB body of
/// 80 texts scored 8160 against the 8192 budget and actually cost 40,960 tokens — 5x the ceiling,
/// ~26s of CPU against a caller that times out at 10s.
///
/// Tokenizing costs microseconds against ~330ms of inference per 512 tokens, so measuring beats
/// estimating. The lock is taken and released here, not held across the batch: `infer` locks per
/// item, and holding it across a whole batch would block the idle-unloader for the batch's lifetime.
fn count_tokens(state: &AppState, texts: &[String]) -> anyhow::Result<usize> {
    let mut guard = state.model.lock().unwrap();
    let model = ensure_loaded(&mut guard, &state.cfg)?;
    let mut total = 0usize;
    for t in texts {
        if is_blank(t) {
            continue; // short-circuited before inference, costs nothing
        }
        let truncated: String = t.chars().take(state.cfg.max_chars).collect();
        let enc = model.tokenizer.encode(truncated, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        // Truncation already caps this at max_tokens; the min is belt and braces.
        total = total.saturating_add(enc.get_ids().len().min(state.cfg.max_tokens));
    }
    Ok(total)
}

// --- HTTP surface (same routes/shapes as server.py) ------------------------

#[derive(Serialize)]
struct HealthResp {
    status: &'static str,
    model: &'static str,
    dims: usize,
    /// Which generation of vectors this serves — see `VECTOR_EPOCH`. This, `dims` and `max_tokens` are
    /// what den-dataset compares; a corpus built under a different epoch cannot be appended to.
    vector_epoch: u32,
    /// Human-readable build identity, for logs and error messages. NOT part of the comparison: it changes
    /// on every release, and treating it as the identity would invalidate a corpus over a log-line fix.
    runtime: String,
    /// Documents are truncated here, so changing it changes the vectors for anything longer.
    max_tokens: usize,
}

#[derive(Serialize)]
struct EmbedResp {
    vector: Vec<i32>,
    dims: usize,
    model: &'static str,
}

#[derive(Serialize)]
struct BatchResp {
    vectors: Vec<Vec<i32>>,
    dims: usize,
    model: &'static str,
}

#[derive(Deserialize)]
struct EmbedQuery {
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct EmbedBody {
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct BatchBody {
    texts: Vec<String>,
}

#[derive(Serialize)]
struct ErrResp {
    detail: String,
}

type AppErr = (StatusCode, Json<ErrResp>);

/// A failing condition repeats on every request — a missing model fails all of them — so each is
/// logged at most this often, with a count of the repeats in between. A timestamp, not a timer.
const FAILURE_LOG_EVERY: Duration = Duration::from_secs(60);

/// The 500 for a failed embed. The real error goes to the log, rate-limited per condition; the text
/// being embedded never does.
fn internal(state: &AppState, e: anyhow::Error) -> AppErr {
    let (condition, detail) = match e.downcast_ref::<LoadFailed>() {
        Some(_) => ("model load", format!("{:#}", e.root_cause())),
        None => ("embedding", format!("{e:#}")),
    };
    let now = Instant::now();
    let mut failures = state.failures.lock().unwrap();
    let (last, repeats) = failures.entry(condition).or_insert((None, 0));
    if last.is_some_and(|at| now.duration_since(at) < FAILURE_LOG_EVERY) {
        *repeats += 1;
    } else {
        *last = Some(now);
        match std::mem::take(repeats) {
            0 => tracing::error!("{condition} failed: {detail}"),
            n => tracing::error!("{condition} failed: {detail} ({n} more since the last line)"),
        }
    }
    (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrResp { detail: "embedding failed".into() }))
}

/// Run `f` on the blocking pool, where all tokenizing and inference happens, turning any failure
/// into the logged 500.
async fn run_blocking<T: Send + 'static>(
    state: &Arc<AppState>,
    f: impl FnOnce(&AppState) -> anyhow::Result<T> + Send + 'static,
) -> Result<T, AppErr> {
    let st = Arc::clone(state);
    match tokio::task::spawn_blocking(move || f(&st)).await {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(internal(state, e)),
        Err(e) => Err(internal(state, e.into())),
    }
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResp> {
    Json(HealthResp {
        status: "ok",
        model: MODEL_LABEL,
        dims: DIMS,
        vector_epoch: VECTOR_EPOCH,
        runtime: format!("den-embed/{}", env!("CARGO_PKG_VERSION")),
        max_tokens: state.cfg.max_tokens,
    })
}

/// `GET /metrics`: Prometheus text format, behind `Authorization: Bearer <METRICS_TOKEN>`.
///
/// With no token configured, or the wrong one given, it answers exactly as an unknown route does — the
/// same JSON 404 — so an install that has not set one is not told there is something here to poke at.
///
/// A scrape is NOT activity, for the same reason `/health` is not: a scraper polling every 15 seconds
/// would otherwise keep ~1.2 GB resident all day, which is precisely what idle-unload exists to
/// prevent. So this only reads — it never calls `touch` or `Model::load` — and everything it reports
/// is state the service already keeps for its own purposes, computed here on request.
async fn metrics(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !metrics_authorized(state.cfg.metrics_token.as_deref(), &headers) {
        return not_found();
    }
    (
        [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8"), (CACHE_CONTROL, "no-store")],
        render_metrics(&state),
    )
        .into_response()
}

/// The answer for an unknown path and for a refused `/metrics`, identical so the two cannot be told
/// apart. The same body every den addon gives.
fn not_found() -> Response {
    (
        StatusCode::NOT_FOUND,
        [(CONTENT_TYPE, "application/json"), (CACHE_CONTROL, "no-store")],
        r#"{"error":"not_found"}"#,
    )
        .into_response()
}

/// Every response is readable cross-origin, and a preflight on any path is answered here, before
/// routing. It returns without reaching a handler, so a preflight never calls `touch` or loads the
/// model — like `/health` and `/metrics`, it is not activity and cannot keep the model warm.
async fn cors(req: Request, next: Next) -> Response {
    if req.method() == Method::OPTIONS {
        return (
            StatusCode::NO_CONTENT,
            [
                (ACCESS_CONTROL_ALLOW_ORIGIN, "*"),
                (ACCESS_CONTROL_ALLOW_METHODS, "GET, HEAD, POST, OPTIONS"),
                (ACCESS_CONTROL_ALLOW_HEADERS, "*"),
                // A day, so a browser stops preflighting every request.
                (ACCESS_CONTROL_MAX_AGE, "86400"),
            ],
        )
            .into_response();
    }
    let mut resp = next.run(req).await;
    resp.headers_mut().insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    resp
}

/// One line per request, written once the response is ready. Only added to the router when
/// `LOG_REQUESTS` is on.
async fn log_request(req: Request, next: Next) -> Response {
    let (method, uri) = (req.method().clone(), req.uri().clone());
    let started = Instant::now();
    let resp = next.run(req).await;
    tracing::info!("{}", request_line(&method, &uri, resp.status(), started.elapsed()));
    resp
}

/// `<METHOD> <path> <status> <ms>ms`, with the path alone: the query of `GET /embed?text=` is the
/// user's search, and it must never reach a log.
fn request_line(method: &Method, uri: &Uri, status: StatusCode, took: Duration) -> String {
    format!("{method} {} {} {}ms", uri.path(), status.as_u16(), took.as_millis())
}

fn metrics_authorized(want: Option<&str>, headers: &HeaderMap) -> bool {
    let Some(want) = want else { return false };
    let given = headers.get(AUTHORIZATION).and_then(|v| v.to_str().ok()).unwrap_or("");
    let given = given.strip_prefix("Bearer ").unwrap_or(given).trim();
    constant_time_eq(given.as_bytes(), want.as_bytes())
}

/// Compare without stopping at the first differing byte, so response time does not reveal how much
/// of a guess was right. A length mismatch returns at once, as Go's `subtle.ConstantTimeCompare`
/// does: that leaks only the token's length, not its contents.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn render_metrics(state: &AppState) -> String {
    // TRY the model lock, never wait on it: inference holds it for the whole of every request (~5 s
    // for a full batch), and a scrape must not queue behind that. Its only holders are inference,
    // which loads the model before anything else, and the unloader's momentary check — so a held
    // lock means the model is in use, and reads as loaded.
    let loaded = match state.model.try_lock() {
        Ok(model) => model.is_some(),
        Err(TryLockError::WouldBlock) => true,
        Err(TryLockError::Poisoned(p)) => p.into_inner().is_some(),
    };
    let (entries, capacity) = {
        let cache = state.cache.lock().unwrap();
        (cache.map.len(), cache.cap)
    };
    // The same arithmetic the unloader does, so this is the number it compares against its limit.
    let now_ms = state.started.elapsed().as_millis() as u64;
    let idle_secs = now_ms.saturating_sub(state.last_used_ms.load(Ordering::Relaxed)) / 1000;
    let unload_secs = state.cfg.idle_unload.map_or(0, |d| d.as_secs());

    format!(
        "# HELP embed_build_info Build identity.
# TYPE embed_build_info gauge
embed_build_info{{version=\"{version}\",model=\"{MODEL_LABEL}\"}} 1
# HELP embed_model_loaded Model resident (1) or idle-unloaded (0).
# TYPE embed_model_loaded gauge
embed_model_loaded {loaded}
# HELP embed_cache_entries Vectors held in the embedding cache.
# TYPE embed_cache_entries gauge
embed_cache_entries {entries}
# HELP embed_cache_capacity Most vectors the embedding cache will hold (0 = cache off).
# TYPE embed_cache_capacity gauge
embed_cache_capacity {capacity}
# HELP embed_idle_seconds Seconds since the last inference, or since boot if there has been none.
# TYPE embed_idle_seconds gauge
embed_idle_seconds {idle_secs}
# HELP embed_idle_unload_seconds Idle time after which the model is unloaded (0 = never).
# TYPE embed_idle_unload_seconds gauge
embed_idle_unload_seconds {unload_secs}
",
        version = env!("CARGO_PKG_VERSION"),
        loaded = u8::from(loaded),
    )
}

async fn embed_get(
    State(state): State<Arc<AppState>>,
    Query(q): Query<EmbedQuery>,
) -> Result<Json<EmbedResp>, AppErr> {
    let vector = run_blocking(&state, move |st| embed_one(st, &q.text)).await?;
    Ok(Json(EmbedResp { vector, dims: DIMS, model: MODEL_LABEL }))
}

async fn embed_post(
    State(state): State<Arc<AppState>>,
    Json(body): Json<EmbedBody>,
) -> Result<Json<EmbedResp>, AppErr> {
    let vector = run_blocking(&state, move |st| embed_one(st, &body.text)).await?;
    Ok(Json(EmbedResp { vector, dims: DIMS, model: MODEL_LABEL }))
}

async fn embed_batch(
    State(state): State<Arc<AppState>>,
    Json(body): Json<BatchBody>,
) -> Result<Json<BatchResp>, AppErr> {
    if body.texts.len() > state.cfg.max_batch {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ErrResp { detail: format!("too many texts (max {})", state.cfg.max_batch) }),
        ));
    }
    // Bound the TOTAL work, not just the count. Each text is separately capped, but the aggregate is
    // what saturates the CPU and queues behind the model — 512 x 8000 chars of CJK measured at
    // ~20-30 minutes of service pinned behind one request.
    //
    // MEASURED, not estimated. Two earlier versions of this check were wrong: counting characters
    // rejected cheap work and admitted expensive work, and `min(chars, max_tokens)` looked sound but
    // is not — the SentencePiece normalizer expands some characters more than 5:1 (see
    // `count_tokens`), which let 5x the budget through.
    let texts_for_count = body.texts.clone();
    let total = run_blocking(&state, move |st| count_tokens(st, &texts_for_count)).await?;
    if total > state.cfg.max_request_tokens {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ErrResp {
                detail: format!(
                    "batch too large: {total} tokens across {} texts (max {} in total)",
                    body.texts.len(),
                    state.cfg.max_request_tokens
                ),
            }),
        ));
    }
    let vectors = run_blocking(&state, move |st| embed_many(st, &body.texts)).await?;
    Ok(Json(BatchResp { vectors, dims: DIMS, model: MODEL_LABEL }))
}

fn spawn_idle_unloader(state: Arc<AppState>, idle: Duration) {
    // Background task: drop the model+tokenizer once idle for `idle`, returning the
    // process to its minimal baseline. The next request lazily reloads them.
    let poll = idle.div_f64(2.0).clamp(Duration::from_secs(15), Duration::from_secs(60));
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(poll).await;
            let idle_ms = idle.as_millis() as u64;
            let now_ms = state.started.elapsed().as_millis() as u64;
            let last = state.last_used_ms.load(Ordering::Relaxed);
            let mut guard = state.model.lock().unwrap();
            if guard.is_some() && now_ms.saturating_sub(last) >= idle_ms {
                *guard = None; // drops Session + Tokenizer → frees to the allocator
                               // ...but glibc keeps freed arenas mapped; hand them back to the OS so
                               // idle RSS actually falls (else it plateaus ~600 MB after unload).
                unsafe {
                    malloc_trim(0);
                }
                tracing::info!("idle-unloaded model after {}s", idle.as_secs());
            }
        }
    });
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // One plain line per event on stderr, the way every den addon logs. No timestamp, level or
    // colour: the systemd journal stamps and names each line itself, and nothing filters on level
    // (the build has no env-filter), so a prefix would only make these lines read unlike the rest.
    tracing_subscriber::fmt()
        .with_target(false)
        .without_time()
        .with_level(false)
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .init();
    // rc.13 returns bool, not Result: false means an environment was already committed, so this
    // config simply does not take effect. Nothing else in the process commits one, and there is no
    // failure to report — but say so rather than discard it silently.
    if !ort::init().with_name("den-embed").commit() {
        tracing::warn!("ort environment was already committed; den-embed's configuration is not in effect");
    }

    let cfg = Config::from_env();
    let addr = format!("0.0.0.0:{}", cfg.port);
    let idle = cfg.idle_unload;
    let cache_max = cfg.cache_max;

    let state = Arc::new(AppState {
        cache: Mutex::new(Lru::new(cache_max)),
        model: Mutex::new(None),
        last_used_ms: AtomicU64::new(0),
        started: Instant::now(),
        failures: Mutex::new(HashMap::new()),
        cfg,
    });

    match idle {
        // Idle-unload on: start unloaded, load lazily on first request.
        Some(d) => spawn_idle_unloader(state.clone(), d),
        // Always-warm: load at boot so the first request isn't cold.
        None => {
            *state.model.lock().unwrap() = Some(Model::load(&state.cfg)?);
        }
    }

    // What this process is running with, secret-free: the metrics token is reported as on or off.
    let on_off = |on: bool| if on { "on" } else { "off" };
    let summary = format!(
        "den-embed {} ({MODEL_LABEL}, idle-unload {}, max_tokens {}, max_request_tokens {}, metrics {}, request log {})",
        env!("CARGO_PKG_VERSION"),
        state.cfg.idle_unload.map_or("off".into(), |d| format!("{}s", d.as_secs())),
        state.cfg.max_tokens,
        state.cfg.max_request_tokens,
        on_off(state.cfg.metrics_token.is_some()),
        on_off(state.cfg.log_requests),
    );
    let app = router(state);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    // Registered BEFORE the readiness line, so nothing is told the service is up while a stop signal
    // would still be a hard kill: until a handler exists SIGTERM keeps its default disposition.
    let shutdown = shutdown_signal();
    let bound = listener.local_addr().map(|a| a.port()).unwrap_or(0);
    // "(port N)" stays last: tests/shutdown.rs reads the bound port from the end of this line.
    tracing::info!("{summary} listening on http://{addr} (port {bound})");

    let outcome = serve_until(listener, app, shutdown, drain_grace()).await;
    match &outcome {
        Outcome::Drained => tracing::info!("shut down cleanly"),
        Outcome::DeadlineHit(why) => tracing::warn!("{why}"),
        Outcome::Failed(why) => tracing::error!("{why}"),
    }
    // EXIT, rather than returning. Returning drops the tokio runtime, and dropping a runtime blocks
    // until every in-flight `spawn_blocking` finishes — which is where all inference runs. So the
    // deadline bounded the drain and then the process sat waiting on the very task it had just given
    // up on: measured 8s deadline, 14.4s actual exit, straight through podman's 10s stop timeout
    // into a SIGKILL, with the response lost anyway. The whole point of the bound is that the stop
    // takes a knowable length of time, and only exiting here delivers that.
    //
    // It also keeps the second-signal escape alive: that is a spawned task, and dropping the runtime
    // cancelled it exactly during the window an operator would be pressing ^C again.
    exit_now(outcome.exit_code());
}

/// The fallback and the CORS layer are added after the routes because a layer only wraps what the
/// router already holds; added before, unknown paths and 405s would go out without the CORS header.
fn router(state: Arc<AppState>) -> Router {
    let max_body = state.cfg.max_body_bytes;
    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/embed", get(embed_get).post(embed_post))
        .route("/embed/batch", post(embed_batch))
        .fallback(|| async { not_found() })
        .layer(axum::extract::DefaultBodyLimit::max(max_body))
        .layer(middleware::from_fn(cors));
    // Outermost, so the logged status is what the client got, preflights included.
    let app = if state.cfg.log_requests { app.layer(middleware::from_fn(log_request)) } else { app };
    app.with_state(state)
}

/// Exit without running `atexit` handlers.
///
/// `std::process::exit` calls libc `exit()`, which runs the statically-linked ONNX Runtime's C++
/// static destructors — while a `Run` may still be executing on a `spawn_blocking` thread. Measured:
/// in 11 of 12 stops that caught inference, the in-flight Run failed with a bogus internal status
/// (`GetElementType is not implemented`, naming a different random node each time) logged as an
/// ERROR indistinguishable from a real inference failure. That is ORT's global state being freed
/// under a running Run — a use-after-free-shaped race that happened to surface as a status. The same
/// measurement with `_exit` produced it 0 times in 6.
///
/// Nothing here needs an atexit handler: the model is read-only, the cache is in-memory, and the log
/// goes to stderr, which is unbuffered, so its lines are already out. Both streams are flushed anyway,
/// because `_exit` will not.
fn exit_now(code: i32) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // SAFETY: `_exit` terminates the process; it has no preconditions.
    unsafe { libc_exit(code) }
}

extern "C" {
    #[link_name = "_exit"]
    fn libc_exit(code: i32) -> !;
}

/// The default drain grace, overridable with `DRAIN_GRACE_SECS`.
///
/// Configurable because it is coupled to the container's stop timeout, which is set outside this
/// binary: raise one and you must raise the other. The default is what is safe with no stop timeout
/// configured at all. The quadlet does set `--stop-timeout=25`, but that file only lands when someone
/// re-runs the provisioner, and the drain must not depend on it having done so.
///
/// The binary is PID 1 in its container (`ENTRYPOINT` exec form), and PID 1 gets no default
/// terminate action — so without a handler SIGTERM was ignored entirely and podman waited its full
/// stop timeout before SIGKILLing: a guaranteed ~10s of downtime on every deploy and auto-update,
/// with every in-flight embed cut. Under podman's DEFAULT 10s rather than the quadlet's 25s, for the
/// reason above; an embed is milliseconds warm and ~1.3s cold, so this is generous.
const DEFAULT_DRAIN_GRACE: Duration = Duration::from_secs(8);

// The default must finish before the smallest external stop timeout that can apply — podman's and
// docker's default 10s, which applies whenever the quadlet has not reached the box. A test only
// guards what someone remembers to run; this fails the build. `MAX_DRAIN_GRACE` is what bounds an
// operator override, since this assert cannot see one.
const _: () = assert!(DEFAULT_DRAIN_GRACE.as_secs() < 10);

/// The largest grace that can still finish before something outside kills us.
///
/// Sized against podman's DEFAULT 10s, not against the quadlet's `--stop-timeout=25`.
///
/// The box now does carry that 25s (it did not for most of the time this was written — `podman
/// inspect` reported 10 while the repo said 25, and a grace of 10..=25 was therefore a value this
/// binary accepted and advertised as safe while being a guaranteed SIGKILL mid-drain). It stays
/// sized to the default anyway: this binary also runs under plain `docker run`, under compose, and
/// on any box provisioned before that line landed, and it cannot see which. A ceiling that is
/// correct everywhere beats one that is correct where someone remembered to re-provision.
///
/// Strictly below, not equal: podman starts its clock at the signal and kills at the timeout, so a
/// grace equal to it loses by however long the deadline takes to fire. Raising this needs the
/// container's stop timeout raised FIRST, and this binary has no way to check that — which is why
/// the conservative number is the one compiled in.
const MAX_DRAIN_GRACE: Duration = Duration::from_secs(9);

// Both the default and the ceiling must finish before the smallest external stop timeout that can
// apply. The default had this assert; the ceiling did not, which is how 25 got in.
const _: () = assert!(MAX_DRAIN_GRACE.as_secs() < 10);
const _: () = assert!(DEFAULT_DRAIN_GRACE.as_secs() <= MAX_DRAIN_GRACE.as_secs());

fn drain_grace() -> Duration {
    let secs = env_clamped(
        "DRAIN_GRACE_SECS",
        DEFAULT_DRAIN_GRACE.as_secs() as usize,
        1,
        MAX_DRAIN_GRACE.as_secs() as usize,
    );
    Duration::from_secs(secs as u64)
}

/// How serving ended. The exit code differs: a drain that ran out of time is expected, a serve
/// error is not.
#[derive(Debug)]
enum Outcome {
    Drained,
    DeadlineHit(String),
    Failed(String),
}

impl Outcome {
    /// A drain that ran out of time is a DESIGNED outcome, so it exits 0. Exiting non-zero would put
    /// the unit into `failed` with Result=exit-code on a routine restart.
    fn exit_code(&self) -> i32 {
        match self {
            Outcome::Drained | Outcome::DeadlineHit(_) => 0,
            Outcome::Failed(_) => 1,
        }
    }
}

/// Serve until `shutdown` resolves, then drain for at most `grace`.
///
/// The bound is the point. `with_graceful_shutdown` waits for every connection task and hyper waits
/// on one that is mid-request, and there is no header-read timeout anywhere — so a client that opens
/// a socket and sends half a request head would hold the process open indefinitely, making restart
/// downtime a function of what an arbitrary client does with a TCP socket.
///
/// `shutdown` is a parameter rather than a direct call so a test can trigger the drain without
/// signalling the test runner itself.
async fn serve_until(
    listener: tokio::net::TcpListener,
    app: Router,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
    grace: Duration,
) -> Outcome {
    let (signalled_tx, signalled_rx) = tokio::sync::oneshot::channel::<()>();
    let serve = axum::serve(listener, app).with_graceful_shutdown(async move {
        shutdown.await;
        let _ = signalled_tx.send(());
    });
    tokio::select! {
        r = serve => match r {
            Ok(()) => Outcome::Drained,
            Err(e) => Outcome::Failed(format!("serve error: {e}")),
        },
        _ = async {
            // The clock starts when the signal ARRIVES, not when the server does — and a DROPPED
            // sender is not an arrival. `oneshot` resolves `Err` immediately when the sender drops,
            // which would start the grace at that instant; today the only path that drops it also
            // completes `serve`, which wins the select, but that is a coincidence to rely on.
            if signalled_rx.await.is_err() {
                std::future::pending::<()>().await
            }
            tokio::time::sleep(grace).await;
        } => Outcome::DeadlineHit(format!("drain deadline ({grace:?}) reached with requests still in flight")),
    }
}

/// Resolves when the process is asked to stop. Handlers are registered eagerly, by the caller.
fn shutdown_signal() -> impl std::future::Future<Output = ()> {
    use tokio::signal::unix::{signal, SignalKind};
    let term = signal(SignalKind::terminate());
    let int = signal(SignalKind::interrupt());
    async move {
        tokio::select! {
            _ = wait_for(term, "SIGTERM") => {}
            _ = wait_for(int, "SIGINT") => {}
        }
        // A SECOND signal ends it now. Both handles are dropped by here and tokio does not restore
        // the default disposition when a `Signal` drops, so without re-registering every later
        // SIGTERM and ^C would be caught and discarded and only SIGKILL would work.
        tokio::spawn(async move {
            tokio::select! {
                _ = quietly(signal(SignalKind::terminate())) => {}
                _ = quietly(signal(SignalKind::interrupt())) => {}
            }
            tracing::warn!("second signal — exiting without finishing the drain");
            // `exit_now`, for the same reason as the deadline path — and this is where the hazard is
            // MOST likely, because you press ^C again precisely when inference is holding the drain
            // open. Measured with `process::exit`: 3 of 3 second-signal stops logged a bogus ORT
            // status naming a random node, against 0 of 2 controls. Fixing one call site and not the
            // other left the worse one behind.
            exit_now(0);
        });
    }
}

/// Resolve when this signal arrives, or NEVER if it could not be registered — resolving immediately
/// would shut the server down the moment it started, which is worse than the hard kill it replaces.
async fn wait_for(registered: std::io::Result<tokio::signal::unix::Signal>, name: &str) {
    match registered {
        Ok(mut sig) => {
            sig.recv().await;
            tracing::info!("{name} — draining in-flight requests");
        }
        Err(e) => {
            tracing::error!("{name} handler unavailable ({e}); it will be a hard kill");
            std::future::pending::<()>().await
        }
    }
}

/// Like `wait_for`, but silent — the caller prints its own, different message.
async fn quietly(registered: std::io::Result<tokio::signal::unix::Signal>) {
    match registered {
        Ok(mut sig) => {
            sig.recv().await;
        }
        Err(_) => std::future::pending::<()>().await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// /health has to distinguish two builds that return DIFFERENT vectors for the same text, because
    /// `model` and `dims` do not: every generation says bge-m3 and 1024. den-dataset records these fields
    /// with the corpus it builds and refuses to append a different identity, so dropping one silently
    /// restores the undetectable drift.
    #[test]
    fn health_reports_what_actually_embedded() {
        let body = serde_json::to_value(HealthResp {
            status: "ok",
            model: MODEL_LABEL,
            dims: DIMS,
            vector_epoch: VECTOR_EPOCH,
            runtime: format!("den-embed/{}", env!("CARGO_PKG_VERSION")),
            max_tokens: 512,
        })
        .unwrap();

        assert_eq!(body["model"], MODEL_LABEL);
        assert_eq!(body["dims"], DIMS);
        assert_eq!(body["max_tokens"], 512);
        assert_eq!(body["vector_epoch"], VECTOR_EPOCH);
        assert_eq!(body["runtime"], format!("den-embed/{}", env!("CARGO_PKG_VERSION")));
        assert_ne!(body["runtime"], "den-embed/");
    }

    /// The epoch must NOT track the crate version. If it did, a release that changed nothing about the
    /// numbers would invalidate every corpus built before it — hours of re-embedding for a log-line fix.
    #[test]
    fn the_vector_epoch_is_not_the_crate_version() {
        assert!(!env!("CARGO_PKG_VERSION").starts_with(&VECTOR_EPOCH.to_string()));
    }

    // Pins the quantization contract (mirrors the Python tests/test_quantize.py).
    // End-to-end byte-parity with the Python service is covered by tests/parity_check.py.
    #[test]
    fn hand_computed_rounding() {
        // Unit vector (0.6, -0.8, 0.0): 0.6*127=76.2->76, -0.8*127=-101.6->-102.
        assert_eq!(quantize_int8(&[0.6, -0.8, 0.0]), vec![76, -102, 0]);
    }

    #[test]
    fn clamp_and_normalize() {
        // Degenerate single axis normalizes to 1.0 -> 127 exactly, no overflow.
        assert_eq!(quantize_int8(&[10.0, 0.0, 0.0]), vec![127, 0, 0]);
    }

    #[test]
    fn zero_vector_stays_zero() {
        assert_eq!(quantize_int8(&[0.0; 8]), vec![0; 8]);
    }

    #[test]
    fn round_half_to_even_matches_numpy() {
        // A pre-normalized value landing exactly on x.5 must round to even (numpy
        // np.round semantics), not away from zero. 0.5/127 normalized back to 0.5.
        assert_eq!(quantize_int8(&[0.5, 0.5]).len(), 2);
    }

    #[test]
    fn blank_detection() {
        assert!(is_blank(""));
        assert!(is_blank("   "));
        assert!(!is_blank("hola"));
    }

    /// The token cap is what bounds activation memory, and `max_chars` does not imply it: at the
    /// 8000-char limit, Hangul and emoji tokenize to ~8000 tokens, and peak RSS measured 1598 MB at
    /// 2048 tokens against a 1536 MB cgroup — a ~10 KB request was an OOM-kill.
    #[test]
    fn the_token_cap_is_below_what_the_memory_limit_allows() {
        let cfg = Config::from_env();
        assert!(cfg.max_tokens <= 1024, "max_tokens {} exceeds what 1536 MB can hold", cfg.max_tokens);
        assert!(cfg.max_tokens >= 16, "max_tokens {} is too small to embed a query", cfg.max_tokens);
    }

    /// Per-text limits bound nothing in aggregate: max_batch x max_chars is 4M characters through a
    /// serial loop that holds the model lock, measured at ~20-30 minutes of pinned service.
    ///
    /// Asserted on the CLAMPS rather than on `Config::from_env()`, because from_env with no
    /// environment set only ever reports the compiled-in defaults — a test that reads like a runtime
    /// guard and is really just documentation. These check that no OPERATOR setting can undo the
    /// bound, which is the part that matters.
    #[test]
    fn no_env_setting_can_undo_the_request_budget() {
        let worst_case_tokens = |max_request_tokens: usize| max_request_tokens;
        // ~0.33s per 512 tokens measured, and den-atlas times this call out at 10s.
        let ceiling = env_clamped("MAX_REQUEST_TOKENS", 8192, 512, 12_288);
        // den-atlas times this call out at 10s. A ceiling that permits more than that is a batch
        // nobody is still waiting for — the earlier 32768 allowed ~21s, double the timeout.
        assert!(
            worst_case_tokens(ceiling) as f64 * 0.33 / 512.0 < 10.0,
            "a legal batch can outlast den-atlas's 10s timeout on this call"
        );

        // And the per-text cap cannot be raised back into the OOM: 1219 MB at 1024 tokens, 1598 MB
        // at 2048, against a 1536 MB cgroup.
        std::env::set_var("MAX_TOKENS", "8192");
        let raised = env_clamped("MAX_TOKENS", 512, 16, 1024);
        std::env::remove_var("MAX_TOKENS");
        assert_eq!(raised, 1024, "an operator could raise the token cap back into the OOM");

        // The ceilings have to hold TOGETHER, not one at a time. Measured: 1219 MB peak at 1024
        // tokens, ~4.2 KB per cache entry, against a 1536 MB cgroup — so both at maximum has to
        // still leave room. Setting each ceiling against the OTHER's default is how 1494 MB got
        // signed off as safe.
        let peak_at_max_tokens_mb = 1219.0;
        let cache_ceiling = env_clamped("CACHE_MAX_ENTRIES", 8192, 0, 32_768);
        let cache_mb = cache_ceiling as f64 * 4.2 / 1024.0;
        assert!(
            peak_at_max_tokens_mb + cache_mb < 1400.0,
            "both ceilings at once is {:.0} MB against a 1536 MB cgroup",
            peak_at_max_tokens_mb + cache_mb
        );
    }

    /// A malformed value must not silently become the opposite of what was asked for.
    /// `IDLE_UNLOAD_SECS='600s'` parsed as nothing and fell back to 0 — always-warm, ~1 GB
    /// resident forever — with no line anywhere saying so.
    #[test]
    fn a_malformed_setting_falls_back_to_the_default_not_to_zero() {
        std::env::set_var("TEST_MALFORMED", "600s");
        assert_eq!(env_clamped("TEST_MALFORMED", 600, 0, 86_400), 600);
        std::env::set_var("TEST_MALFORMED", "");
        assert_eq!(env_clamped("TEST_MALFORMED", 600, 0, 86_400), 600);
        std::env::set_var("TEST_MALFORMED", "-5");
        assert_eq!(env_clamped("TEST_MALFORMED", 600, 0, 86_400), 600);
        std::env::set_var("TEST_MALFORMED", " 42 ");
        assert_eq!(env_clamped("TEST_MALFORMED", 600, 0, 86_400), 42, "a padded number is still a number");
        std::env::remove_var("TEST_MALFORMED");
    }

    /// A state as `main` builds it with idle-unload on — model unloaded, idle clock never reset — but a
    /// minute old, so anything that reset the idle clock would move it visibly.
    fn metrics_state(token: Option<&str>) -> Arc<AppState> {
        let mut cfg = Config::from_env();
        cfg.metrics_token = token.map(Into::into);
        cfg.idle_unload = Some(Duration::from_secs(600));
        Arc::new(AppState {
            cache: Mutex::new(Lru::new(cfg.cache_max)),
            model: Mutex::new(None),
            last_used_ms: AtomicU64::new(0),
            started: Instant::now() - Duration::from_secs(60),
            failures: Mutex::new(HashMap::new()),
            cfg,
        })
    }

    /// `GET /embed?text=` carries the user's search in the query, so the line has the path alone.
    #[test]
    fn the_request_log_never_carries_the_query() {
        let uri: Uri = "/embed?text=a%20private%20search".parse().unwrap();
        let line = request_line(&Method::GET, &uri, StatusCode::OK, Duration::from_millis(12));
        assert_eq!(line, "GET /embed 200 12ms");
        assert!(!line.contains("private"), "{line}");
    }

    async fn scrape(state: Arc<AppState>, auth: Option<&str>) -> (StatusCode, String, String) {
        let mut headers = HeaderMap::new();
        if let Some(auth) = auth {
            headers.insert(AUTHORIZATION, auth.parse().unwrap());
        }
        let resp = metrics(State(state), headers).await;
        let status = resp.status();
        let content_type =
            resp.headers().get(CONTENT_TYPE).map(|v| v.to_str().unwrap().to_string()).unwrap_or_default();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        (status, content_type, String::from_utf8(body.to_vec()).unwrap())
    }

    /// Unset means OFF, and indistinguishable from a route that does not exist — not an empty 200.
    #[tokio::test]
    async fn metrics_is_not_found_without_a_configured_token() {
        for auth in [None, Some("Bearer "), Some("Bearer anything")] {
            let (status, content_type, body) = scrape(metrics_state(None), auth).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "served metrics with no token configured ({auth:?})");
            assert_eq!(content_type, "application/json");
            assert_eq!(body, r#"{"error":"not_found"}"#, "unlike an unknown route's 404");
        }
    }

    async fn call(state: Arc<AppState>, method: Method, uri: &str) -> Response {
        use tower::ServiceExt;
        let req =
            axum::http::Request::builder().method(method).uri(uri).body(axum::body::Body::empty()).unwrap();
        router(state).oneshot(req).await.unwrap()
    }

    fn header<'a>(resp: &'a Response, name: &str) -> &'a str {
        resp.headers().get(name).map(|v| v.to_str().unwrap()).unwrap_or_default()
    }

    /// A browser preflights any path it means to call, including ones with no OPTIONS route, and
    /// answering one must not count as use: that would keep ~1.2 GB resident for anything that
    /// preflights on a timer, which is what idle-unload exists to prevent.
    #[tokio::test]
    async fn a_preflight_on_any_path_is_answered_without_waking_the_model() {
        let state = metrics_state(None);
        for path in ["/embed", "/embed/batch", "/health", "/metrics", "/nope"] {
            let resp = call(state.clone(), Method::OPTIONS, path).await;
            assert_eq!(resp.status(), StatusCode::NO_CONTENT, "{path}");
            assert_eq!(header(&resp, "access-control-allow-origin"), "*", "{path}");
            assert_eq!(header(&resp, "access-control-allow-methods"), "GET, HEAD, POST, OPTIONS", "{path}");
            assert_eq!(header(&resp, "access-control-allow-headers"), "*", "{path}");
            assert_eq!(header(&resp, "access-control-max-age"), "86400", "{path}");
        }
        assert_eq!(state.last_used_ms.load(Ordering::Relaxed), 0, "a preflight reset the idle clock");
        assert!(state.model.lock().unwrap().is_none(), "a preflight loaded the model");
    }

    /// Success, a refused route, an unknown path and a wrong method alike: a response without the
    /// header is unreadable to a browser, whatever its status says.
    #[tokio::test]
    async fn every_response_is_readable_cross_origin() {
        for (method, path, status) in [
            (Method::GET, "/health", StatusCode::OK),
            (Method::GET, "/embed?text=", StatusCode::OK), // blank: a zero vector, no model needed
            (Method::GET, "/metrics", StatusCode::NOT_FOUND),
            (Method::GET, "/nope", StatusCode::NOT_FOUND),
            (Method::DELETE, "/health", StatusCode::METHOD_NOT_ALLOWED),
        ] {
            let resp = call(metrics_state(None), method.clone(), path).await;
            assert_eq!(resp.status(), status, "{method} {path}");
            assert_eq!(header(&resp, "access-control-allow-origin"), "*", "{method} {path}");
        }
    }

    #[tokio::test]
    async fn an_unknown_path_is_a_json_404() {
        let resp = call(metrics_state(None), Method::GET, "/nope").await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        assert_eq!(header(&resp, "content-type"), "application/json");
        assert_eq!(header(&resp, "cache-control"), "no-store");
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        assert_eq!(&body[..], br#"{"error":"not_found"}"#);
    }

    #[tokio::test]
    async fn metrics_is_not_found_with_the_wrong_token() {
        for auth in [None, Some("Bearer wrong"), Some("Bearer s3cre"), Some("Bearer s3cret2")] {
            let (status, _, _) = scrape(metrics_state(Some("s3cret")), auth).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{auth:?} was accepted");
        }
    }

    #[tokio::test]
    async fn metrics_serves_prometheus_text_with_the_right_token() {
        let (status, content_type, body) = scrape(metrics_state(Some("s3cret")), Some("Bearer s3cret")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/plain; version=0.0.4; charset=utf-8");
        let build_info =
            format!("embed_build_info{{version=\"{}\",model=\"bge-m3\"}} 1\n", env!("CARGO_PKG_VERSION"));
        assert!(body.contains(&build_info), "no build_info line in:\n{body}");
        assert!(body.contains("\nembed_idle_unload_seconds 600\n"), "{body}");
        assert!(body.contains("# TYPE embed_model_loaded gauge\n"), "{body}");
    }

    /// A scrape must not keep the model warm: counting it as activity would pin ~1.2 GB resident for
    /// as long as anything polls, and loading the model to report on it would be worse.
    #[tokio::test]
    async fn a_scrape_is_not_activity_and_does_not_load_the_model() {
        let state = metrics_state(Some("s3cret"));
        let (_, _, body) = scrape(state.clone(), Some("Bearer s3cret")).await;

        assert_eq!(state.last_used_ms.load(Ordering::Relaxed), 0, "the scrape reset the idle clock");
        assert!(state.model.lock().unwrap().is_none(), "the scrape loaded the model");
        assert!(body.contains("\nembed_model_loaded 0\n"), "{body}");
        let idle: u64 = body
            .lines()
            .find_map(|l| l.strip_prefix("embed_idle_seconds "))
            .and_then(|v| v.parse().ok())
            .expect("no embed_idle_seconds sample");
        assert!(idle >= 60, "idle clock reads {idle}s on a service untouched for a minute");
    }

    #[test]
    fn constant_time_eq_is_plain_equality() {
        assert!(constant_time_eq(b"s3cret", b"s3cret"));
        assert!(!constant_time_eq(b"s3cret", b"s3creT"));
        assert!(!constant_time_eq(b"s3cret", b"s3cre"));
        assert!(!constant_time_eq(b"", b"s3cret"));
    }

    #[test]
    fn cache_key_is_deterministic_and_model_scoped() {
        let a = cache_key("hola");
        assert_eq!(a, cache_key("hola"));
        assert_ne!(a, cache_key("adios"));
        // blake2b-128 -> 32 hex chars.
        assert_eq!(a.len(), 32);
    }
}
