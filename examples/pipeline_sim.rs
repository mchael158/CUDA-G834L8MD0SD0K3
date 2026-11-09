use corewarp::{BackendKind, CoreWarp, Operation, Priority, SchedulingPolicy};

fn main() -> corewarp::Result<()> {
    let mut rt = CoreWarp::builder()
        .backend(BackendKind::Host)
        .streams(3)
        .issue_width(2)
        .max_in_flight(32)
        .scheduling(SchedulingPolicy::Fixed)
        .build()?;

    println!("backend: {}", rt.backend_name());

    const N: usize = 1 << 20;
    let x = rt.upload(&vec![1.0; N])?;
    let y = rt.upload(&vec![2.0; N])?;

    let handle = rt.submit_with_priority(
        Priority::High,
        Operation::Saxpy {
            alpha: 3.0,
            x,
            y,
        },
    )?;

    // Mais três ops independentes (vector add / scale / reduction).
    let a = rt.upload(&vec![1.0; 1024])?;
    let b = rt.upload(&vec![2.0; 1024])?;
    let out = rt.alloc_zeros(1024)?;
    rt.submit(Operation::VectorAdd { a, b, output: out })?;

    let inp = rt.upload(&vec![2.0; 1024])?;
    let scaled = rt.alloc_zeros(1024)?;
    rt.submit(Operation::VectorScale {
        alpha: 4.0,
        input: inp,
        output: scaled,
    })?;

    let red_in = rt.upload(&vec![1.0; 100])?;
    let red_out = rt.alloc_zeros(1)?;
    rt.submit(Operation::ReductionSum {
        input: red_in,
        output: red_out,
    })?;

    rt.wait(&handle)?;
    rt.drain()?;

    let y_host = rt.download(y)?;
    let add_host = rt.download(out)?;
    let scale_host = rt.download(scaled)?;
    let sum_host = rt.download(red_out)?;

    assert!((y_host[0] - 5.0).abs() < 1e-3);
    assert!((add_host[0] - 3.0).abs() < 1e-3);
    assert!((scale_host[0] - 8.0).abs() < 1e-3);
    assert!((sum_host[0] - 100.0).abs() < 1e-3);

    let m = rt.metrics();
    println!(
        "submitted={} completed={} polls={} pool_hits={} pool_misses={} y[0]={:.1}",
        m.jobs_submitted, m.jobs_completed, m.polls, m.pool_hits, m.pool_misses, y_host[0]
    );

    rt.shutdown()?;
    println!("pipeline_sim ok");
    Ok(())
}
