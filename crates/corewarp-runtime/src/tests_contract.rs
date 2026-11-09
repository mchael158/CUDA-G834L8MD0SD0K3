//! Host-runtime contract tests (no CUDA required).
//!
//! These tests pin the 0.1 behavioral guarantees that application packs depend on:
//! correct op results, priority ordering under a single slot, shape rejection,
//! download waiting for completion, pool reuse after free, and serialization of
//! overlapping writable buffers. Run with:
//! `cargo test -p corewarp-runtime --release`.

use corewarp_core::{BackendKind, Error, Operation, Priority, SchedulingPolicy};

use crate::CoreWarp;

/// Build a small Fixed host runtime used by most tests.
fn host_rt() -> CoreWarp {
    CoreWarp::builder()
        .backend(BackendKind::Host)
        .streams(3)
        .issue_width(2)
        .max_in_flight(8)
        .scheduling(SchedulingPolicy::Fixed)
        .build()
        .unwrap()
}

/// SAXPY, VectorAdd, VectorScale, and ReductionSum must match host reference math.
#[test]
fn saxpy_vector_ops_and_reduction() {
    let mut rt = host_rt();
    let x = rt.upload(&[1.0, 1.0, 1.0, 1.0]).unwrap();
    let y = rt.upload(&[2.0, 2.0, 2.0, 2.0]).unwrap();
    let h = rt
        .submit(Operation::Saxpy { alpha: 3.0, x, y })
        .unwrap();
    rt.wait(&h).unwrap();
    // y = 2 + 3*1 = 5
    assert_eq!(rt.download(y).unwrap(), vec![5.0, 5.0, 5.0, 5.0]);

    let a = rt.upload(&[1.0, 2.0]).unwrap();
    let b = rt.upload(&[3.0, 4.0]).unwrap();
    let o = rt.alloc_zeros(2).unwrap();
    rt.submit(Operation::VectorAdd { a, b, output: o })
        .unwrap();
    rt.drain().unwrap();
    assert_eq!(rt.download(o).unwrap(), vec![4.0, 6.0]);

    let inp = rt.upload(&[2.0, 3.0]).unwrap();
    let out = rt.alloc_zeros(2).unwrap();
    rt.submit(Operation::VectorScale {
        alpha: 4.0,
        input: inp,
        output: out,
    })
    .unwrap();
    rt.drain().unwrap();
    assert_eq!(rt.download(out).unwrap(), vec![8.0, 12.0]);

    let red_in = rt.upload(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let red_out = rt.alloc_zeros(1).unwrap();
    rt.submit(Operation::ReductionSum {
        input: red_in,
        output: red_out,
    })
    .unwrap();
    rt.drain().unwrap();
    assert_eq!(rt.download(red_out).unwrap(), vec![10.0]);
}

/// Inclusive prefix-sum: [1,2,3,4] → [1,3,6,10].
#[test]
fn prefix_sum_inclusive() {
    let mut rt = host_rt();
    let input = rt.upload(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let output = rt.alloc_zeros(4).unwrap();
    rt.submit(Operation::PrefixSum { input, output })
        .unwrap();
    rt.drain().unwrap();
    assert_eq!(rt.download(output).unwrap(), vec![1.0, 3.0, 6.0, 10.0]);
}

/// Histogram over [0, 10) with 5 bins of width 2.
#[test]
fn histogram_bins() {
    let mut rt = host_rt();
    // Values in [0, 10) → 5 bins of width 2:
    // 0.0,1.5 → bin0; 2.0 → bin1; 4.9 → bin2; 9.9 → bin4
    let input = rt.upload(&[0.0, 1.5, 2.0, 4.9, 9.9]).unwrap();
    let output = rt.alloc_zeros(5).unwrap();
    rt.submit(Operation::Histogram {
        input,
        output,
        min_v: 0.0,
        max_v: 10.0,
    })
    .unwrap();
    rt.drain().unwrap();
    assert_eq!(rt.download(output).unwrap(), vec![2.0, 1.0, 1.0, 0.0, 1.0]);
}

/// With a single slot, High priority must be issued before Low.
#[test]
fn priority_high_emitted_before_low() {
    let mut rt = CoreWarp::builder()
        .backend(BackendKind::Host)
        .streams(1) // force serialized issue
        .issue_width(1)
        .max_in_flight(1)
        .scheduling(SchedulingPolicy::Fixed)
        .build()
        .unwrap();

    let low_y = rt.upload(&[0.0]).unwrap();
    let high_y = rt.upload(&[0.0]).unwrap();
    let x = rt.upload(&[1.0]).unwrap();

    let low = rt
        .submit_with_priority(
            Priority::Low,
            Operation::Saxpy {
                alpha: 1.0,
                x,
                y: low_y,
            },
        )
        .unwrap();
    let high = rt
        .submit_with_priority(
            Priority::High,
            Operation::Saxpy {
                alpha: 1.0,
                x,
                y: high_y,
            },
        )
        .unwrap();

    rt.poll().unwrap();
    // With 1 slot: only High should be Running/Completed; Low still Queued.
    use corewarp_core::JobStatus;
    assert_ne!(rt.status(&low).unwrap(), JobStatus::Running);
    assert!(matches!(
        rt.status(&high).unwrap(),
        JobStatus::Running | JobStatus::Completed
    ));
    rt.drain().unwrap();
    assert!(rt.is_complete(&high).unwrap());
    assert!(rt.is_complete(&low).unwrap());
}

/// Mismatched lengths must be rejected at submit time (no partial enqueue).
#[test]
fn rejects_bad_shapes() {
    let mut rt = host_rt();
    let x = rt.upload(&[1.0, 2.0]).unwrap();
    let y = rt.upload(&[1.0]).unwrap();
    let err = rt
        .submit(Operation::Saxpy { alpha: 1.0, x, y })
        .unwrap_err();
    assert!(matches!(err, Error::Invalid(_)));

    let input = rt.upload(&[1.0, 2.0, 3.0]).unwrap();
    let output = rt.alloc_zeros(2).unwrap();
    let err = rt
        .submit(Operation::ReductionSum { input, output })
        .unwrap_err();
    assert!(matches!(err, Error::Invalid(_)));
}

/// `download` must internally poll until the buffer is no longer in-flight.
#[test]
fn download_waits_until_complete() {
    let mut rt = host_rt();
    let x = rt.upload(&vec![1.0; 1024]).unwrap();
    let y = rt.upload(&vec![0.0; 1024]).unwrap();
    rt.submit(Operation::Saxpy { alpha: 2.0, x, y })
        .unwrap();
    // download polls internally until in_flight == 0
    let out = rt.download(y).unwrap();
    assert!((out[0] - 2.0).abs() < 1e-5);
}

/// Freeing a Small-class buffer then re-uploading the same size should hit the pool.
#[test]
fn pool_reuses_after_free() {
    let mut rt = host_rt();
    let a = rt.upload(&vec![1.0; 64]).unwrap(); // Small class
    rt.free_buffer(a).unwrap();
    let b = rt.upload(&vec![2.0; 64]).unwrap();
    let m = rt.metrics();
    assert!(
        m.pool_hits >= 1,
        "expected a pool hit after free+upload of the same size class"
    );
    let _ = b;
}

/// Two SAXPYs writing the same `y` must serialize so the final result is 2.0, not a race.
#[test]
fn serializes_overlapping_writable_buffers() {
    let mut rt = host_rt();
    let x = rt.upload(&[1.0; 8]).unwrap();
    let y = rt.upload(&[0.0; 8]).unwrap();

    // Two SAXPYs on the same y: must serialize (no corruption).
    rt.submit(Operation::Saxpy { alpha: 1.0, x, y })
        .unwrap();
    rt.submit(Operation::Saxpy { alpha: 1.0, x, y })
        .unwrap();
    rt.drain().unwrap();
    let out = rt.download(y).unwrap();
    // First: y = 0 + 1*x = 1; second: y = 1 + 1*x = 2
    assert!((out[0] - 2.0).abs() < 1e-5);
}
