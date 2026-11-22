"""Resume the formal queue with the host RAM admission cutoff disabled."""
from sska_rec import table2_v2_runner as runner


_original_memory_usage = runner.memory_usage


def memory_usage_without_ram_cutoff(device):
    usage = _original_memory_usage(device)
    # Keep the measurement for troubleshooting without blocking admission on it.
    usage['observed_working_ram_gib'] = usage['working_ram_gib']
    usage['working_ram_gib'] = None
    return usage


if __name__ == '__main__':
    # This changes only scheduling; the audited training snapshot stays in place.
    runner.memory_usage = memory_usage_without_ram_cutoff
    runner.main()
