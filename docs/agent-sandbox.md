# Agent sandbox: what is hardened and what is not

Bots run in pooled, pre-started worker processes (`arena/agents/worker.py`,
`arena/agents/_worker_main.py`) confined by the layers below
(`arena/agents/sandbox.py`). Everything uses features this VM's kernel provides to an
unprivileged user. **gVisor, containers, cgroups and seccomp-notify supervisors are
not used.** This is a reasonable practice sandbox for running your own or semi-trusted
code. It is not a hardened multi-tenant boundary for hostile code.

Measured on this VM: Linux 6.12, 8 vCPUs (KVM guest, Intel Xeon), 47 GiB RAM,
Landlock ABI 6.

## Layers, in the order they are applied

1. **Namespaces (util-linux `unshare`).** Each worker starts with
   `unshare --map-current-user --net --ipc --uts --pid --fork --kill-child --mount --mount-proc`.
   - New user, network, IPC, UTS, mount and PID namespaces.
   - The worker runs as the invoking unprivileged user. It is not root in the
     namespace, because only the current uid is mapped.
   - The network namespace has only a loopback interface, which is down. There is no
     route to the host or the internet.
   - The worker is PID 1 of its own PID namespace. `/proc` is remounted, so it sees only
     itself and cannot signal the parent.
   - The worker and its helper are in their own process group, so a kill removes both.
2. **Trusted warm-up.** The worker imports the SDK, the domain models and pydantic.
   It validates a synthetic observation 200 times to warm the code paths, then calls
   `gc.freeze()` once the bot has loaded. No bot code has run yet.
3. **Resource limits (`setrlimit`).**

   | Limit | Value |
   | --- | --- |
   | `RLIMIT_AS` | 1 GiB address space |
   | `RLIMIT_CPU` | 300 s per worker (one table) |
   | `RLIMIT_FSIZE` | 0, so no file can be written |
   | `RLIMIT_NOFILE` | 64 |
   | `RLIMIT_NPROC` | threads + 2 = 10 (counted per user namespace), allowing at most 8 bot threads |
   | `RLIMIT_CORE`, `RLIMIT_MEMLOCK`, `RLIMIT_MSGQUEUE` | 0 |

4. **`PR_SET_NO_NEW_PRIVS`** and **`PR_SET_DUMPABLE=0`**.
5. **Landlock.**
   - First layer, at start-up: read-only access to the Python installation, the
     virtualenv's site-packages, and the bot staging root. Nothing anywhere is writable
     or executable.
   - TCP bind and connect are denied (ABI 4 and later).
   - Signals and abstract UNIX sockets outside the worker's domain are denied (ABI 6).
   - Second layer, at load: read access narrows further to the runtime paths plus this
     bot's own bundle (`staging_root/<sha256>/`). Bundles are copied there and made
     read-only.
   - The repository itself is not readable, so a bot can import only the arena modules
     that were already loaded in step 2.
6. **seccomp-BPF (x86-64).**
   - Wrong-architecture and x32 syscalls kill the process.
   - These syscalls are denied with `EPERM`: socket creation, connect, bind, listen and
     accept; fork and vfork; exec; ptrace and `process_vm_*`; `pidfd_open` and
     `pidfd_getfd`; mount and namespace changes, chroot and pivot_root; bpf,
     perf_event_open and userfaultfd; io_uring; the keyring calls; module loading,
     kexec and reboot; and SysV and POSIX IPC creation.
   - `clone` is allowed only with `CLONE_THREAD`, so threads work and processes do not.
     `clone3` returns `ENOSYS`, so glibc falls back to `clone`.
7. **Interpreter and I/O.**
   - The worker runs `python -s -P -B` with `cwd=/`.
   - The environment is minimal: `PATH`, `LANG` and `PYTHONHASHSEED=0`.
   - `sys.path` is filtered to readable entries.
   - Requests and replies use two inherited pipes with length-prefixed frames. The
     parent caps reply frames at 16 KiB of metadata and 64 KiB of state.
   - stdout and stderr go to a separate non-blocking pipe, capped at 64 KiB. The
     parent keeps the first 4 KiB.

If any layer fails to apply, the worker reports `sandbox_error` and exits rather than
running with a weaker sandbox. `tests/agents/test_sandbox.py` runs a probe bot inside a
real worker and asserts each denial:

- network sockets and connects,
- reading `/etc/passwd`, the repository and `$HOME`,
- writing to `/tmp` and to the bot's own directory,
- fork, subprocess and exec,
- signalling the parent,
- the `/proc` view,
- SysV and POSIX shared memory,
- the thread cap,
- the uid, PID 1, the environment, and loopback as the only interface.

`tests/agents/test_worker_faults.py` covers illegal, malformed, raising, crashing,
looping, slow, memory-hungry, chatty and oversized bots, plus a bot that forges protocol
frames on stdout. `tests/agents/test_t12.py` covers spec T12.

## Not hardened (known limitations)

- **No gVisor or VM boundary.** Workers share the host kernel. A kernel exploit that is
  reachable through the allowed syscalls (most of them) escapes the sandbox. The seccomp
  filter is a **deny list**, not an allow list. It exists only for x86-64. On other
  architectures, sandboxed mode refuses to start.
- **No cgroups.** Memory is bounded by `RLIMIT_AS` (address space, not resident memory),
  and there is no cgroup memory, CPU-share or PID controller.
  - Many workers that each touch close to 1 GiB could still pressure host memory.
  - CPU is bounded only by the per-worker `RLIMIT_CPU` and by the deadline. A worker
    that spins between decisions is killed at the seat's next decision (after the
    20 ms grace period). Until then it competes for CPU.
- **Same uid as the parent.** Isolation from the parent's files relies on Landlock,
  and isolation from its processes relies on the PID namespace plus Landlock signal
  scoping. Both are kernel features enforced by the same kernel.
- **Allowed reads.** The Python installation and site-packages are readable. A bot can
  import any installed package and read its files, but nothing else on the host.
- **Timing side channels are not addressed.** Workers share CPUs and caches with other
  workers and the parent.
- **Crashes can look like timeouts.** The `unshare` helper holds the worker's pipe ends
  until it reaps the worker, and tearing down a large address space takes time. So a
  worker that dies mid-decision is sometimes reported as `timeout` rather than
  `worker_crash`. Both fall back the same way, and the worker is replaced either way.
  Only the reason label differs.
- **Instance-state bots lose memory on restart.** This is by design. Such seats are
  flagged `restorable: false`.
- **Bot-reported timings are untrusted.** `worker_wall_ns` and `worker_cpu_ns` come from
  the worker and are reported only for information. Deadlines use only the parent's
  clock.

## Deadline accuracy on this VM

`received_ns` is a userspace `perf_counter_ns()` taken after the read that completes
the reply. If the parent is descheduled, a reply that was already waiting is judged
late. That is conservative against the bot, never in its favour. To keep that rare:

- the parent busy-polls the reply pipe during the budget,
- workers are warmed before they report ready,
- workers spin for up to 3 ms after the wake notice that precedes each request (sent
  before the clock starts, carrying no game data). This hides the roughly 250 µs vCPU
  wake-up latency of this KVM guest.

Baselines in sandboxed mode, 2 parallel jobs (the old default; the current default
is 1 job on this VM, see below), NLHE, budget 1,000,000 ns. Times are end to end as
the parent sees them.

| Run | Decisions | p50 | p95 | p99 | max | 1 ms fallbacks |
| --- | --- | --- | --- | --- | --- | --- |
| 6 seats, 240 tables | 222,354 | 0.000176 s | 0.000281 s | 0.000379 s | 0.004870 s | 307 (0.138%) |
| 8 seats, 160 tables | 177,128 | 0.000183 s | 0.000291 s | 0.000390 s | 0.010378 s | 227 (0.128%) |

The 8-seat maximum happened while a separate worker start-up measurement was running on
the same VM.

- The bots' own `decide()` wall time was p50 about 12–26 µs and p99 under 0.1 ms.
- Nearly all of the round trip is serialization, the pipe hop, validation of the
  observation in the worker, and scheduling.
- The fallbacks are tail scheduling events on a shared VM. They are not slow bots.
- One check_fold worker in the 6-seat run did not answer within the 20 ms grace period
  after a late reply. It was killed and restored from committed state, and the table
  finished normally.

**More parallel jobs is worse.** Every decision needs two busy-polling processes (the
parent and the worker). On 8 vCPUs, the 1 ms timeout rate measured in tuning was:

| Jobs | 1 ms timeout rate |
| --- | --- |
| 1 | about 0.1% |
| 2 | about 0.1% |
| 3 | 2.8% |
| 4 | 13.6% |

Later measurements with strategy population lineups found 0.02% at 1 job against
0.12–0.24% at 2 jobs, pinned or not. With 2 jobs the timeouts cluster in the first
75 decisions of each table, while replacement workers start. So the default is
**1 job per 4 physical cores**, after reserving 1 core for the driving process
(`arena.benchmark.capacity.safe_jobs`). That is 1 job on this 8-core VM, 3 on 16
and 7 on 32. It counts physical cores, not SMT threads, and honours CPU affinity and
cgroup v2 quotas. `--jobs` overrides it. Timeout rates also rise when other heavy
work shares the VM (for example a test run). Results from a loaded machine say more
about the machine than about the bot. Each manifest records the process layout and
the measured timeout rate.

**Host stalls and the classified timeout gate.** Even at 1 job on an idle VM, a
calling station misses the deadline on 0.035–0.04% of decisions. The cause is the
host: about 4–6 vCPU stalls of 1 ms or more per minute, and kvm-clock with an
unstable TSC. The stalls barely show in steal time. Two things keep the harness
itself out of those windows:

- the parent's garbage collector is off from dispatch until the reply is read;
- `play_table` freezes the existing heap for the duration of the table.

Every timeout becomes a `TimeoutRecord` with the worker's own times from the late
reply: parse wall and CPU (`worker_parse_ns`, `worker_parse_cpu_ns`), decide wall
and CPU (`worker_wall_ns`, `worker_cpu_ns`), the garbage collections that ran inside
the parse (`worker_parse_gc`) and the worker's Python thread count when it replied
(`worker_threads`). `SeatStats.worker_parse_ns` keeps every decision's parse time in
order (-1 for a timeout or a parse that ran a collection), so a timeout's parse can be
compared with the seat's median and with its neighbours. With `timeout_inputs` the observation
and committed state are kept as well, so `replay_decision` can replay the exact
decision in a fresh worker. `run_benchmark(control=..., control_every=k)` interleaves
tables seated entirely with a control bot, so a run measures its own stall rate.

The stalls are charged to thread CPU time. A probe bot that spends exactly 160 µs
of thread CPU time per decision (6 sandboxed seats, 86,394 decisions) timed out 19
times, and every late reply showed CPU within 10% of wall. One decide reported
1.74 ms of wall and 1.74 ms of CPU for a 160 µs loop, and two parses reported
1.0–1.2 ms of CPU against a usual 130 µs. CPU ≈ wall therefore does not separate a
host stall from bot computation here. The host also slows the CPU without stopping
it: in gate control records the harness parse (fixed work) ran 3–7 times its usual
time with CPU ≈ wall. A bot that stops after a fixed CPU time is immune to that, so
the control burner does a fixed amount of work instead.

`strategy.eval.stalls` builds the gate on top of these hooks, with a workload-matched
burner (`strategy.eval.burn`) as the control:

- bot-caused timeouts (a CPU-bound decide past the deadline that a host slowdown
  does not explain, a slow replay, or a late reply from a worker with bot threads)
  count against a 0.02% ceiling and against the control's own bot-caused timeouts;
- stall-like timeouts are tested against the control's (C-test);
- the control must stay under a 0.05% ceiling;
- up to 2 reruns are allowed, and every attempt is reported.

## Throughput

- A 6-seat sandboxed table takes about 1 s of wall time (roughly 900 decisions).
- 240 tables with 2 jobs take about 2 minutes (the default for `match` is 1 job on
  this VM, about twice as long).
- Worker start-up (namespaces, interpreter, imports, warm-up) is about 0.14 s. It runs
  in a background pool, off the table's critical path.
