//! Shutdown behaviour, against the real binary.
//!
//! These cannot be unit tests: the paths they cover end in `std::process::exit`, and a signal is
//! process-wide, so exercising them in-process would take down the test runner.
//!
//! den-embed runs as PID 1 in its container (`ENTRYPOINT` exec form), and PID 1 gets no default
//! terminate action — before this, SIGTERM was ignored outright and podman SIGKILLed after its stop
//! timeout, cutting every in-flight embed on every deploy and auto-update.
//!
//! `IDLE_UNLOAD_SECS=1` keeps the model unloaded, so these start in well under a second and
//! never touch the 555 MB ONNX file.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const SIGTERM: i32 = 15;
const SIGINT: i32 = 2;

/// `child.wait()` with a deadline, and a kill if it blows through it.
///
/// A bare `wait()` turns "shutdown is broken" into "the test suite hangs forever and leaks a
/// den-embed per test" — which is exactly what happened on a build where the drain did not work:
/// no failure, no output, four orphaned children holding a model each. In CI it would burn the job
/// timeout instead of going red. A test for shutdown must never depend on shutdown working.
fn wait_within(child: &mut Child, limit: Duration) -> Option<std::process::ExitStatus> {
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) => std::thread::sleep(Duration::from_millis(25)),
            Err(e) => panic!("try_wait failed: {e}"),
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    None
}

fn signal(child: &Child, sig: i32) {
    extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    unsafe { kill(child.id() as i32, sig) };
}

/// Everything the server printed, shared with the draining thread.
type Log = std::sync::Arc<std::sync::Mutex<String>>;

/// The bound port from the readiness line, `den-embed <version> listening on :<port> — …`.
fn listening_port(line: &str) -> Option<u16> {
    line.split("listening on :").nth(1)?.split_whitespace().next()?.parse().ok()
}

/// Start den-embed on an ephemeral port and wait until it reports the port it bound.
fn start() -> (Child, u16) {
    let (c, p, _) = start_logged(&[]);
    (c, p)
}

fn start_with(extra: &[(&str, &str)]) -> (Child, u16) {
    let (c, p, _) = start_logged(extra);
    (c, p)
}

/// Same, but keeping the server's output — the only way to assert on something it says rather than
/// on timing. The ambient environment is cleared: the knobs carry no prefix to filter on, and a
/// developer with `DRAIN_GRACE_SECS` exported would otherwise silently change several tests' timing.
fn start_logged(extra: &[(&str, &str)]) -> (Child, u16, Log) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_den-embed"));
    cmd.env_clear()
        .env("PORT", "0")
        // Lazy-load, so no model file is ever opened.
        .env("IDLE_UNLOAD_SECS", "1")
        .env("MODEL_DIR", "/nonexistent-on-purpose")
        // The log, readiness line included, goes to STDERR.
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    // AFTER the defaults, so a caller's override wins rather than being silently replaced.
    for (k, v) in extra {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().expect("the binary must start");

    let stderr = child.stderr.take().unwrap();
    let mut reader = BufReader::new(stderr);
    let mut port = None;
    let mut seen = String::new();
    for _ in 0..20 {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        seen.push_str(&line);
        if let Some(p) = listening_port(&line) {
            port = Some(p);
            break;
        }
    }
    let port: u16 = port.expect("the binary never reported a listening port");
    // Keep draining stderr so the pipe cannot fill and block the child, and keep what it said.
    let log: Log = std::sync::Arc::new(std::sync::Mutex::new(seen));
    let sink = log.clone();
    let drained = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let done = drained.clone();
    std::thread::spawn(move || {
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            sink.lock().unwrap().push_str(&line);
            line.clear();
        }
        // EOF: the child closed stderr, so everything it ever wrote is now in the buffer.
        done.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    DRAINED.lock().unwrap().push(drained);
    (child, port, log)
}

/// EOF flags for the log-draining threads, so a test can wait for one rather than racing it.
static DRAINED: std::sync::Mutex<Vec<std::sync::Arc<std::sync::atomic::AtomicBool>>> =
    std::sync::Mutex::new(Vec::new());

/// The server's full output, once its stderr has actually reached EOF.
///
/// Reading the buffer straight after `wait_within` races the draining thread: the last line is
/// written microseconds before `_exit` while the poll interval is 25ms, so under load the assertion
/// would fail with the maximally confusing "no inference was in flight".
fn log_after_exit(log: &Log) -> String {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let all_done = DRAINED.lock().unwrap().iter().all(|d| d.load(std::sync::atomic::Ordering::SeqCst));
        if all_done || Instant::now() > deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    log.lock().unwrap().clone()
}

/// A client holding half a request head must not hold shutdown open indefinitely. Nothing in the
/// stack times out a partial head, so without a deadline one socket decides how long the service is
/// down — and the model can take a while to drop, making the window worse.
#[test]
fn a_partial_request_cannot_hold_shutdown_open() {
    let (mut child, port) = start();
    let mut stuck = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stuck.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n").unwrap(); // no blank line
    stuck.flush().unwrap();
    std::thread::sleep(Duration::from_millis(200));

    let started = Instant::now();
    signal(&child, SIGTERM);
    let status = wait_within(&mut child, Duration::from_secs(20))
        .expect("the process never exited — the drain is unbounded");
    let took = started.elapsed();

    assert!(took > Duration::from_secs(4), "it did not drain at all: {took:?}");
    assert!(took < Duration::from_secs(12), "the drain is unbounded: {took:?}");
    // A deadline reached is designed, not a crash: exiting non-zero puts the unit in `failed`, and
    // any client can cause it.
    assert_eq!(status.code(), Some(0), "a routine drain timeout reported a crash");
}

/// A second signal ends the drain at once. tokio does not restore the default disposition when a
/// `Signal` drops, so without re-registering, every later SIGTERM and ^C is caught and discarded and
/// only SIGKILL works.
#[test]
fn a_second_signal_ends_the_drain_at_once() {
    for sig in [SIGTERM, SIGINT] {
        let (mut child, port) = start();
        let mut stuck = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        stuck.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n").unwrap();
        stuck.flush().unwrap();
        std::thread::sleep(Duration::from_millis(200));

        signal(&child, SIGTERM);
        std::thread::sleep(Duration::from_millis(300));
        let started = Instant::now();
        signal(&child, sig);
        let status = wait_within(&mut child, Duration::from_secs(20))
            .expect("the process never exited after a second signal");

        assert!(
            started.elapsed() < Duration::from_secs(3),
            "a second signal ({sig}) did not end the drain: {:?}",
            started.elapsed()
        );
        assert_eq!(status.code(), Some(0), "asking twice is deliberate, not a failure");
    }
}

/// An idle service exits at once — the grace costs nothing when nothing is in flight. Signalling
/// immediately after the readiness line also pins that handlers are registered BEFORE it: they used
/// to register lazily, so a stop in that window killed the process outright (exit by signal, no code).
#[test]
fn an_idle_service_exits_at_once() {
    let (mut child, _port) = start();
    let started = Instant::now();
    signal(&child, SIGTERM);
    let status = wait_within(&mut child, Duration::from_secs(20)).expect("an idle process never exited");
    assert!(started.elapsed() < Duration::from_secs(3), "idle exit took {:?}", started.elapsed());
    assert_eq!(status.code(), Some(0), "killed by signal rather than handled");
}

/// An in-flight request that finishes inside the grace must COMPLETE, not be cut. /health is served
/// without the model, so this exercises the drain rather than inference.
#[test]
fn an_in_flight_request_completes_across_a_stop() {
    let (mut child, port) = start();
    let mut sock = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    // Send a complete request, let the server accept and start on it, THEN signal before reading.
    // Without the pause the signal can land before the connection is even accepted, and hyper closes
    // it as idle — which proves nothing about draining.
    sock.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").unwrap();
    sock.flush().unwrap();
    std::thread::sleep(Duration::from_millis(150));
    signal(&child, SIGTERM);

    let mut body = String::new();
    let mut r = BufReader::new(sock);
    let mut line = String::new();
    while r.read_line(&mut line).unwrap_or(0) > 0 {
        body.push_str(&line);
        line.clear();
    }
    assert!(body.contains("200 OK"), "the in-flight response was cut: {body:?}");
    assert!(body.contains("\"status\""), "no body came back: {body:?}");
    let status = wait_within(&mut child, Duration::from_secs(20)).expect("the process never exited");
    assert_eq!(status.code(), Some(0));
}

/// An over-large grace is clamped, and says so.
///
/// The ceiling is what keeps the drain finishing before something outside kills us — podman's
/// default is 10s and the box's container still reports `StopTimeout=10` — so a grace above it is
/// not a longer drain, it is a SIGKILL mid-drain. The clamp shipped without a test: deleting it
/// from `drain_grace()` left the whole suite passing. This costs no wall clock, because it asserts
/// on the warning rather than waiting out a drain.
#[test]
fn an_over_large_drain_grace_is_clamped() {
    let (mut child, _port, log) = start_logged(&[("DRAIN_GRACE_SECS", "600")]);
    signal(&child, SIGTERM);
    let status = wait_within(&mut child, Duration::from_secs(20)).expect("never exited");
    assert_eq!(status.code(), Some(0));

    let out = log_after_exit(&log);
    assert!(
        out.contains("is outside") && out.contains("using 9"),
        "600s was accepted as a drain grace; the ceiling is not enforced. Server said:\n{out}"
    );
}

/// The grace clock starts at the SIGNAL, not at server start.
///
/// A clock started at boot would deadline INSTANTLY on any service that has been up longer than the
/// grace — cutting the drain to zero, which is worse than not having one. The comment in `main.rs`
/// asserts this property; nothing tested it, and the mutation survived the whole suite.
#[test]
fn the_grace_clock_starts_at_the_signal_not_at_boot() {
    let (mut child, port) = start_with(&[("DRAIN_GRACE_SECS", "2")]);

    // Well past the grace, with no signal sent. The service must still be serving.
    std::thread::sleep(Duration::from_secs(4));
    let mut probe = TcpStream::connect(("127.0.0.1", port)).expect("still accepting");
    probe.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").unwrap();
    let mut body = String::new();
    let mut r = BufReader::new(probe);
    let mut line = String::new();
    while r.read_line(&mut line).unwrap_or(0) > 0 {
        body.push_str(&line);
        line.clear();
    }
    assert!(
        body.contains("200 OK"),
        "the service stopped serving during ordinary uptime — the grace clock started at boot"
    );

    // ...and it still drains normally afterwards.
    let started = Instant::now();
    signal(&child, SIGTERM);
    let status = wait_within(&mut child, Duration::from_secs(20)).expect("never exited");
    assert!(started.elapsed() < Duration::from_secs(3), "took {:?}", started.elapsed());
    assert_eq!(status.code(), Some(0));
}

/// The stop must be bounded even with INFERENCE in flight — which is the only case that matters,
/// and the one every test above misses.
///
/// Inference runs in `spawn_blocking`. Returning from `main` drops the tokio runtime, and dropping a
/// runtime blocks until every in-flight blocking task finishes — so the deadline fired at 8s and the
/// process then sat waiting on the very request it had given up on: measured 14.4s, straight through
/// podman's 10s stop timeout into a SIGKILL, with the response lost anyway. Exiting instead of
/// returning is what makes the bound real.
///
/// Needs the real model, so it is skipped unless `TEST_MODEL_DIR` points at a directory
/// holding `model_int8.onnx` + `tokenizer.json`. Skipped rather than faked: nothing smaller than the
/// real model produces a blocking task long enough to tell the two behaviours apart.
/// `#[ignore]`, not a silent early return. Returning made CI report `6 passed; 0 ignored` — green,
/// with the one test guarding the exit-vs-return regression having asserted nothing, and its
/// explanatory `eprintln!` swallowed without `--nocapture`. An absent test must not be
/// indistinguishable from a passing one. Run it with:
///
///     TEST_MODEL_DIR=<dir> cargo test --test shutdown -- --ignored
#[test]
#[ignore = "needs the 555 MB model; set TEST_MODEL_DIR"]
fn inference_in_flight_does_not_extend_the_stop() {
    let model_dir = std::env::var("TEST_MODEL_DIR")
        .expect("set TEST_MODEL_DIR to a dir with model_int8.onnx + tokenizer.json");

    let mut child = Command::new(env!("CARGO_BIN_EXE_den-embed"))
        .env("PORT", "0")
        .env("MODEL_DIR", &model_dir)
        .env("IDLE_UNLOAD_SECS", "0") // always-warm: load at boot, so the request is the slow part
        // A SHORT grace, so the deadline fires well before the batch finishes. At the 8s default the
        // batch completed first, the deadline never fired, and the test passed against the bug.
        .env("DRAIN_GRACE_SECS", "2")
        // Clamped to the 12288 ceiling, which still admits this batch whole: measured 7.35s of work
        // on an M-series laptop, so it is still running well past the grace.
        .env("MAX_REQUEST_TOKENS", "32768")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the binary must start");

    let stderr = child.stderr.take().unwrap();
    let mut reader = BufReader::new(stderr);
    let mut port = None;
    for _ in 0..40 {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        if let Some(p) = listening_port(&line) {
            port = Some(p);
            break;
        }
    }
    let port = port.expect("no listening port");
    // KEEP the server's output. The deadline line is the only reliable evidence that a blocking task
    // was actually in flight: the batch's own HTTP response cannot say so, because it does not
    // arrive until inference finishes, which is after the stop. An earlier version asserted on a
    // flag set by the response and could never have been true.
    let log = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = log.clone();
    std::thread::spawn(move || {
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            sink.lock().unwrap().push_str(&line);
            line.clear();
        }
    });

    // Sized to the REQUEST BUDGET, not to max_batch. The first version of this sent 512 long texts
    // and was rejected with 413 in 2ms by the very budget added in the same change — before
    // spawn_blocking, before any inference — so the deadline branch was never taken and the test
    // passed against the reverted bug. It asserted nothing.
    //
    // Every property of this payload is load-bearing, and each was found by the test failing:
    //  - CJK, because the budget estimates one token per character and only CJK realizes it. English
    //    is ~4 chars/token, so an English batch sized to the budget finished in 0.3s.
    //  - VARIED characters, because a repeated character collapses under BPE into a handful of
    //    tokens: 480 identical Hangul syllables are not 480 tokens.
    //  - UNIQUE per text, because identical texts are content-cache hits — 16 identical texts is one
    //    inference and 15 lookups, which also finished in 0.3s.
    // At the default 8192-token budget the work tops out around 2.5s, too close to any grace worth
    // testing, so the run below raises the budget (a supported setting; the 32768 it asks for is
    // clamped to the 12288 ceiling in env_clamped).
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let texts: Vec<String> = (0..64)
        .map(|i| {
            let body: String =
                (0..500).map(|_| char::from_u32(0xAC00 + (next() % 11172) as u32).unwrap()).collect();
            format!("{body}{i}")
        })
        .collect();
    let body = format!("{{\"texts\":{}}}", serde_json_stub(&texts));
    let addr = format!("127.0.0.1:{port}");
    std::thread::spawn(move || {
        if let Ok(mut s) = TcpStream::connect(&addr) {
            let req = format!(
                "POST /embed/batch HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = s.write_all(req.as_bytes());
            let _ = s.flush();
            let mut sink = Vec::new();
            let _ = std::io::Read::read_to_end(&mut s, &mut sink);
        }
    });
    std::thread::sleep(Duration::from_millis(1500)); // let the batch get into spawn_blocking

    let started = Instant::now();
    signal(&child, SIGTERM);
    let status = wait_within(&mut child, Duration::from_secs(30)).expect("the process never exited");
    let took = started.elapsed();

    // The stop must land close to the grace, not grace + however long the blocking task had left.
    // With the bug it was grace + the remainder of the batch, which is unbounded.
    assert!(
        took < Duration::from_secs(5),
        "stop took {took:?} against a 2s grace — the blocking task extended it, so the bound is not real"
    );
    assert_eq!(status.code(), Some(0), "exited by signal or with a failure code");

    // ...and the deadline is what ended it, which only happens with work still running. Without
    // this, a batch rejected as too large produces the same fast, clean stop and the test passes
    // against the very bug it exists to catch — which is exactly what an earlier version did.
    let out = log_after_exit(&log);
    assert!(
        out.contains("drain deadline"),
        "the stop was clean, so no inference was in flight and this proves nothing. Server said:\n{out}"
    );
}

/// Minimal JSON string-array encoder, so the test needs no serde dependency of its own.
fn serde_json_stub(texts: &[String]) -> String {
    let parts: Vec<String> = texts.iter().map(|t| format!("\"{}\"", t.replace('"', "\\\""))).collect();
    format!("[{}]", parts.join(","))
}
