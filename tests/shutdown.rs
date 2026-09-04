//! Shutdown behaviour, against the real binary.
//!
//! These cannot be unit tests: the paths they cover end in `std::process::exit`, and a signal is
//! process-wide, so exercising them in-process would take down the test runner.
//!
//! den-embed runs as PID 1 in its container (`ENTRYPOINT` exec form), and PID 1 gets no default
//! terminate action — before this, SIGTERM was ignored outright and podman SIGKILLed after its stop
//! timeout, cutting every in-flight embed on every deploy and auto-update.
//!
//! `DEN_EMBED_IDLE_UNLOAD_SEC=1` keeps the model unloaded, so these start in well under a second and
//! never touch the 555 MB ONNX file.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const SIGTERM: i32 = 15;
const SIGINT: i32 = 2;

fn signal(child: &Child, sig: i32) {
    extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    unsafe { kill(child.id() as i32, sig) };
}

/// Start den-embed on an ephemeral port and wait until it reports the port it bound.
fn start() -> (Child, u16) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_den-embed"))
        .env("DEN_EMBED_PORT", "0")
        .env("DEN_EMBED_HOST", "127.0.0.1")
        // Lazy-load, so no model file is ever opened.
        .env("DEN_EMBED_IDLE_UNLOAD_SEC", "1")
        .env("DEN_EMBED_MODEL_DIR", "/nonexistent-on-purpose")
        // tracing_subscriber::fmt() writes to STDOUT, so that is where the readiness line is.
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("the binary must start");

    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);
    let mut port = None;
    for _ in 0..20 {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        if let Some(rest) = line.split("(port ").nth(1) {
            port = rest.trim_end().trim_end_matches(')').parse().ok();
            break;
        }
    }
    let port: u16 = port.expect("the binary never reported a listening port");
    // Keep draining stdout so the pipe cannot fill and block the child.
    std::thread::spawn(move || {
        let mut sink = String::new();
        while reader.read_line(&mut sink).unwrap_or(0) > 0 {
            sink.clear();
        }
    });
    (child, port)
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
    let status = child.wait().expect("wait");
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
        let status = child.wait().expect("wait");

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
    let status = child.wait().expect("wait");
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
    let status = child.wait().expect("wait");
    assert_eq!(status.code(), Some(0));
}
