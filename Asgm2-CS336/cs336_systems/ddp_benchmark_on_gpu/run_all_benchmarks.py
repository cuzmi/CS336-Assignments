import os
import subprocess
import sys


MODULES = [
    "cs336_systems.ddp_benchmark_on_gpu.benchmark_naive_ddp",
    "cs336_systems.ddp_benchmark_on_gpu.benchmark_naive_ddp_flat",
    "cs336_systems.ddp_benchmark_on_gpu.benchmark_naive_ddp_async",
    "cs336_systems.ddp_benchmark_on_gpu.benchmark_naive_bucket",
    "cs336_systems.ddp_benchmark_on_gpu.profile_ddp_peak_memory",
]


def main():
    python_exe = sys.executable
    shared_env = os.environ.copy()

    for module_name in MODULES:
        print("\n" + "=" * 80)
        print(f"RUNNING: {module_name}")
        print("=" * 80)
        sys.stdout.flush()

        subprocess.run(
            [python_exe, "-m", module_name],
            check=True,
            env=shared_env,
        )


if __name__ == "__main__":
    main()
