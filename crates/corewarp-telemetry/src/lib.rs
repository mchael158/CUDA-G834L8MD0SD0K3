#[derive(Debug, Clone, Copy, Default)]
pub struct JobMetrics {
    pub queue_wait_us: u64,
    pub h2d_us: u64,
    pub kernel_us: u64,
    pub d2h_us: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RuntimeMetrics {
    pub jobs_submitted: u64,
    pub jobs_completed: u64,
    pub jobs_failed: u64,
    pub bytes_h2d: u64,
    pub bytes_d2h: u64,
    pub polls: u64,
    pub pool_hits: u64,
    pub pool_misses: u64,
    pub last_gpu_ms: f64,
}
