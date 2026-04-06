(base) root@ubuntu22:~/CS336-Assignments/Asgm2-CS336/cs336_systems/ddp_benchmark_on_gpu# AUTO_NSYS=1 NSYS_TRACE=cuda,nvtx BENCHMARK_WARMUP=0 BENCHMARK_ITERS=1 python -m cs336_systems.ddp_benchmark_on_gpu.run_all_benchmarks

================================================================================
RUNNING: cs336_systems.ddp_benchmark_on_gpu.benchmark_naive_ddp
================================================================================
[rank 0] train_one_step=0.947077s, all_reduce_time=0.571030s, ratio=60.294%
[rank 0] train_one_step=0.942502s, all_reduce_time=0.566000s, ratio=60.053%
[rank 0] train_one_step=0.944198s, all_reduce_time=0.567535s, ratio=60.108%
[rank 0] train_one_step=0.943401s, all_reduce_time=0.568893s, ratio=60.302%
[rank 0] train_one_step=0.946075s, all_reduce_time=0.571382s, ratio=60.395%

=== naive_ddp_benchmark ===
rank 0: avg_train_one_step=0.944651s
rank 1: avg_train_one_step=0.944704s
max_across_ranks: avg_train_one_step=0.944704s
rank 0: avg_all_reduce_time=0.568968s
rank 1: avg_all_reduce_time=0.569380s
max_across_ranks: avg_all_reduce_time=0.569380s
rank 0: avg_ratio_pct=60.231%
rank 1: avg_ratio_pct=60.271%
max_across_ranks: avg_ratio_pct=60.271%
Generating '/tmp/nsys-report-4f6d.qdstrm'
[1/1] [========================100%] naive_ddp_nsys.nsys-rep
Generated:
    /root/CS336-Assignments/Asgm2-CS336/cs336_systems/nsys_reports/naive_ddp_nsys.nsys-re

================================================================================
RUNNING: cs336_systems.ddp_benchmark_on_gpu.benchmark_naive_ddp_flat
================================================================================
[rank 0] train_one_step=0.935370s, all_reduce_time=0.558882s, ratio=59.750%
[rank 0] train_one_step=0.932312s, all_reduce_time=0.557625s, ratio=59.811%
[rank 0] train_one_step=0.933988s, all_reduce_time=0.559177s, ratio=59.870%
[rank 0] train_one_step=0.934558s, all_reduce_time=0.558826s, ratio=59.796%
[rank 0] train_one_step=0.931833s, all_reduce_time=0.556572s, ratio=59.729%

=== flat_ddp_benchmark ===
rank 0: avg_train_one_step=0.933612s
rank 1: avg_train_one_step=0.933648s
max_across_ranks: avg_train_one_step=0.933648s
rank 0: avg_all_reduce_time=0.558216s
rank 1: avg_all_reduce_time=0.560096s
max_across_ranks: avg_all_reduce_time=0.560096s
rank 0: avg_ratio_pct=59.791%
rank 1: avg_ratio_pct=59.990%
max_across_ranks: avg_ratio_pct=59.990%

================================================================================
RUNNING: cs336_systems.ddp_benchmark_on_gpu.benchmark_naive_ddp_async
================================================================================
[rank 0] train_one_step=0.852794s, sync_time=0.017575s, ratio=2.061%
[rank 0] train_one_step=0.855397s, sync_time=0.017541s, ratio=2.051%
[rank 0] train_one_step=0.852781s, sync_time=0.017521s, ratio=2.055%
[rank 0] train_one_step=0.853217s, sync_time=0.017660s, ratio=2.070%
[rank 0] train_one_step=0.854095s, sync_time=0.017575s, ratio=2.058%

=== overlap_individual_parameters_benchmark ===
rank 0: avg_train_one_step=0.853657s
rank 1: avg_train_one_step=0.853693s
max_across_ranks: avg_train_one_step=0.853693s
rank 0: avg_sync_time=0.017574s
rank 1: avg_sync_time=0.017550s
max_across_ranks: avg_sync_time=0.017574s
rank 0: avg_ratio_pct=2.059%
rank 1: avg_ratio_pct=2.056%
max_across_ranks: avg_ratio_pct=2.059%

================================================================================
RUNNING: cs336_systems.ddp_benchmark_on_gpu.benchmark_naive_bucket
================================================================================
[bucket_size: 1][rank: 0] train_one_step = 0.845590s,wait_time = 0.005135s,ratio = 0.607%
[bucket_size: 1][rank: 0] train_one_step = 0.843986s,wait_time = 0.005139s,ratio = 0.609%
[bucket_size: 1][rank: 0] train_one_step = 0.843439s,wait_time = 0.005056s,ratio = 0.599%
[bucket_size: 1][rank: 0] train_one_step = 0.844395s,wait_time = 0.005067s,ratio = 0.600%
[bucket_size: 1][rank: 0] train_one_step = 0.847802s,wait_time = 0.005079s,ratio = 0.599%

=== bucketed_ddp_benchmark_bucket_1mb ===
rank 0: avg_train_one_step=0.845042s
rank 1: avg_train_one_step=0.845078s
max_across_ranks: avg_train_one_step=0.845078s
rank 0: avg_wait_time=0.005095s
rank 1: avg_wait_time=0.005093s
max_across_ranks: avg_wait_time=0.005095s
rank 0: avg_ratio_pct=0.603%
rank 1: avg_ratio_pct=0.603%
max_across_ranks: avg_ratio_pct=0.603%
[bucket_size: 10][rank: 0] train_one_step = 0.837868s,wait_time = 0.005068s,ratio = 0.605%
[bucket_size: 10][rank: 0] train_one_step = 0.834209s,wait_time = 0.004951s,ratio = 0.593%
[bucket_size: 10][rank: 0] train_one_step = 0.839951s,wait_time = 0.005225s,ratio = 0.622%
[bucket_size: 10][rank: 0] train_one_step = 0.833482s,wait_time = 0.005071s,ratio = 0.608%
[bucket_size: 10][rank: 0] train_one_step = 0.840303s,wait_time = 0.005071s,ratio = 0.603%

=== bucketed_ddp_benchmark_bucket_10mb ===
rank 0: avg_train_one_step=0.837163s
rank 1: avg_train_one_step=0.837202s
max_across_ranks: avg_train_one_step=0.837202s
rank 0: avg_wait_time=0.005077s
rank 1: avg_wait_time=0.005105s
max_across_ranks: avg_wait_time=0.005105s
rank 0: avg_ratio_pct=0.606%
rank 1: avg_ratio_pct=0.610%
max_across_ranks: avg_ratio_pct=0.610%
[bucket_size: 100][rank: 0] train_one_step = 0.872036s,wait_time = 0.017909s,ratio = 2.054%
[bucket_size: 100][rank: 0] train_one_step = 0.866776s,wait_time = 0.017880s,ratio = 2.063%
[bucket_size: 100][rank: 0] train_one_step = 0.866700s,wait_time = 0.017877s,ratio = 2.063%
[bucket_size: 100][rank: 0] train_one_step = 0.864906s,wait_time = 0.017910s,ratio = 2.071%
[bucket_size: 100][rank: 0] train_one_step = 0.866547s,wait_time = 0.017867s,ratio = 2.062%

=== bucketed_ddp_benchmark_bucket_100mb ===
rank 0: avg_train_one_step=0.867393s
rank 1: avg_train_one_step=0.867403s
max_across_ranks: avg_train_one_step=0.867403s
rank 0: avg_wait_time=0.017889s
rank 1: avg_wait_time=0.017917s
max_across_ranks: avg_wait_time=0.017917s
rank 0: avg_ratio_pct=2.062%
rank 1: avg_ratio_pct=2.066%
max_across_ranks: avg_ratio_pct=2.066%
[bucket_size: 1000][rank: 0] train_one_step = 0.883591s,wait_time = 0.018158s,ratio = 2.055%
[bucket_size: 1000][rank: 0] train_one_step = 0.883576s,wait_time = 0.018195s,ratio = 2.059%
[bucket_size: 1000][rank: 0] train_one_step = 0.870172s,wait_time = 0.018152s,ratio = 2.086%
[bucket_size: 1000][rank: 0] train_one_step = 0.888280s,wait_time = 0.018145s,ratio = 2.043%
[bucket_size: 1000][rank: 0] train_one_step = 0.876186s,wait_time = 0.018191s,ratio = 2.076%

=== bucketed_ddp_benchmark_bucket_1000mb ===
rank 0: avg_train_one_step=0.880361s
rank 1: avg_train_one_step=0.880425s
max_across_ranks: avg_train_one_step=0.880425s
rank 0: avg_wait_time=0.018168s
rank 1: avg_wait_time=0.018168s
max_across_ranks: avg_wait_time=0.018168s
rank 0: avg_ratio_pct=2.064%
rank 1: avg_ratio_pct=2.064%
max_across_ranks: avg_ratio_pct=2.064%

================================================================================
RUNNING: cs336_systems.ddp_benchmark_on_gpu.profile_ddp_peak_memory
================================================================================

=== mode: ddp ===
[average train time]
rank 0: avg_train_one_step=0.949837s
rank 1: avg_train_one_step=0.949829s
max_across_ranks: avg_train_one_step=0.949837s
[after building model]
rank 0: allocated_mb=7804.68, peak_allocated_mb=7804.68
rank 1: allocated_mb=7804.68, peak_allocated_mb=7804.68
max_across_ranks: allocated_mb=7804.68, peak_allocated_mb=7804.68
[before optimizer step]
rank 0: allocated_mb=31237.98, peak_allocated_mb=31299.98
rank 1: allocated_mb=31237.98, peak_allocated_mb=31299.98
max_across_ranks: allocated_mb=31237.98, peak_allocated_mb=31299.98
[after optimizer step]
rank 0: allocated_mb=31237.98, peak_allocated_mb=39042.13
rank 1: allocated_mb=31237.98, peak_allocated_mb=39042.13
max_across_ranks: allocated_mb=31237.98, peak_allocated_mb=39042.13

=== mode: sharded ===
[average train time]
rank 0: avg_train_one_step=1.196870s
rank 1: avg_train_one_step=1.196854s
max_across_ranks: avg_train_one_step=1.196870s
[after building model]
rank 0: allocated_mb=7804.68, peak_allocated_mb=7804.68
rank 1: allocated_mb=7804.68, peak_allocated_mb=7804.68
max_across_ranks: allocated_mb=7804.68, peak_allocated_mb=7804.68
[before optimizer step]
rank 0: allocated_mb=23557.36, peak_allocated_mb=23619.36
rank 1: allocated_mb=23307.50, peak_allocated_mb=23369.50
max_across_ranks: allocated_mb=23557.36, peak_allocated_mb=23619.36
[after optimizer step]
rank 0: allocated_mb=23557.36, peak_allocated_mb=27521.20
rank 1: allocated_mb=23307.50, peak_allocated_mb=27147.34
max_across_ranks: allocated_mb=23557.36, peak_allocated_mb=27521.20