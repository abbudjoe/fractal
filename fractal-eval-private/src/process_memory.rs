use std::mem::MaybeUninit;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProcessMemoryMetricKind {
    #[default]
    PeakRss,
    PhysicalFootprint,
}

impl ProcessMemoryMetricKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PeakRss => "peak_rss",
            Self::PhysicalFootprint => "physical_footprint",
        }
    }
}

pub fn process_memory_metric_kind() -> ProcessMemoryMetricKind {
    #[cfg(target_os = "macos")]
    {
        ProcessMemoryMetricKind::PhysicalFootprint
    }
    #[cfg(not(target_os = "macos"))]
    {
        ProcessMemoryMetricKind::PeakRss
    }
}

pub fn process_memory_measurement_note(sample_window: &str) -> String {
    match process_memory_metric_kind() {
        ProcessMemoryMetricKind::PhysicalFootprint => format!(
            "process memory metrics are sampled {sample_window} via proc_pid_rusage(RUSAGE_INFO_V4).ri_phys_footprint; on macOS this tracks physical footprint and unified-memory pressure more faithfully than RSS"
        ),
        ProcessMemoryMetricKind::PeakRss => format!(
            "process memory metrics are sampled {sample_window} via getrusage(RUSAGE_SELF).ru_maxrss; this tracks peak resident memory for process-level trend detection"
        ),
    }
}

pub fn process_peak_memory_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        process_peak_physical_footprint_bytes().unwrap_or_else(process_peak_rss_bytes)
    }
    #[cfg(not(target_os = "macos"))]
    {
        process_peak_rss_bytes()
    }
}

#[cfg(target_os = "macos")]
fn process_peak_physical_footprint_bytes() -> Option<u64> {
    let mut usage = MaybeUninit::<libc::rusage_info_v4>::uninit();
    let status = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            libc::RUSAGE_INFO_V4,
            usage.as_mut_ptr().cast(),
        )
    };
    if status != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    Some(usage.ri_phys_footprint)
}

fn process_peak_rss_bytes() -> u64 {
    let mut usage = MaybeUninit::<libc::rusage>::uninit();
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return 0;
    }
    let usage = unsafe { usage.assume_init() };
    #[cfg(target_os = "macos")]
    {
        usage.ru_maxrss as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        (usage.ru_maxrss as u64) * 1024
    }
}

#[cfg(test)]
mod tests {
    use super::{
        process_memory_measurement_note, process_memory_metric_kind, process_peak_memory_bytes,
        ProcessMemoryMetricKind,
    };

    #[test]
    fn process_memory_sampler_reports_a_nonzero_value() {
        let bytes = process_peak_memory_bytes();
        assert!(bytes > 0);
    }

    #[test]
    fn process_memory_note_matches_platform_metric() {
        let note = process_memory_measurement_note("during tests");
        match process_memory_metric_kind() {
            ProcessMemoryMetricKind::PeakRss => assert!(note.contains("ru_maxrss")),
            ProcessMemoryMetricKind::PhysicalFootprint => {
                assert!(note.contains("ri_phys_footprint"))
            }
        }
    }
}
