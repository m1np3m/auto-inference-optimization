# Inference optimization harness: requirements (v1)

Status: requirements, agreed 2026-10-14. Next step: design (plugin interfaces, run-record schema,
tracker mapping, campaign config format).

## 1. Purpose

Bring the autoresearch idea (edit, run under a fixed protocol, keep or discard, repeat) to
**inference engines**. The core of this project is not the search loop but the **referee**: a
hardware-agnostic measurement and evidence harness that answers, with evidence:

> Did change X make engine Y better on workload Z, without hurting quality, and how do we know?

The improvement loop (a human in v1, an agent in v2) sits on top of the referee and never changes it.

| autoresearch | this harness |
|---|---|
| `train.py` (agent edits) | an **arm**: engine config (flags, env, tuned configs) |
| `prepare.py` (read-only) | the **referee**: launch, gates, load, metrics, scoring, evidence |
| `val_bpb` | **goodput at an SLO**, quality as a hard gate |
| 5-minute fixed budget | fixed **evaluation protocol** (workload, seeds, warm-up, window, reps) |
| `results.tsv` + git keep/reset | **tracker** runs with full lineage |
| `program.md` (human edits) | campaign agenda and rules (v2) |

## 2. Principles

- **Infra-agnostic.** Nothing in the core names a vendor, cluster or node. Everything environment-
  specific is a plugin chosen in a config file.
- **Measurement first.** A wrong number is worse than no number; every number carries its noise floor
  and its provenance.
- **Referee is enforced, not requested.** The thing being optimized can never edit the thing that
  scores it.
- **Graceful degradation.** A plugin declares its capabilities; a missing capability is recorded as
  missing, never silently skipped.

## 3. Plugin points

| Plugin | Responsibility | Example (reference profile) | Alternatives |
|---|---|---|---|
| Engine launcher | start / health / stop an engine from an arm; map its metrics to canonical names; flag allowlist | vLLM in Docker | SGLang, TRT-LLM, local process, k8s |
| Workload source | requests + arrival pattern + data card | captured production conversations | ShareGPT, synthetic random |
| Load generator | drive the workload, record every request | **Locust replay** | `vllm bench serve`, genai-perf, k6 |
| Quality evaluator | hard gate and robustness score | token-match + **tf-eval** | lm-eval, custom judge |
| Profiler | on-demand kernel / CPU / system traces | vLLM torch profiler, rocprofv3 | nsys / ncu, py-spy, perf |
| Telemetry collector | engine + hardware time series | engine `/metrics` scrape, rocm-smi | nvidia-smi / DCGM, node-exporter |
| Tracker | params, metrics, artifacts, lineage | **MLflow** | W&B, files + git |
| Worker | where an arm runs (one GPU, node, instance) and its isolation | a GPU slot with a CPU slice | cloud instance, laptop |

v1 ships the vLLM launcher only, behind the generic interface.

## 4. Data

- **D1 Versioned workloads.** Content hash, source, capture date and a data card: input / output length
  distributions, shared-prefix ratio, tool-call share, thinking share, turn count. Runs reference the hash.
- **D2 Arrival pattern is part of the workload.** Each workload declares **open-loop** (original
  timestamps, optional time scaling, or a rate) or **closed-loop** (N concurrent users). Both are
  supported; the SLO sweep is closed-loop, the realism replay is open-loop.
- **D3 Real vs synthetic.** Replayed real traces decide keep / discard. Synthetic random data is for
  sweeps and saturation only (it misrepresents speculative-decoding acceptance and prefix-cache hits).
- **D4 Fixed quality set.** A fixed prompt set with stored reference outputs per baseline (for token
  match), and a pinned tf-eval suite version.
- **D5 Privacy.** Captured traces may hold customer data: redaction step and an access label; the
  harness refuses to upload unlabeled traces to the tracker.

## 5. Measurement methodology (the referee)

- **M1 Warm-up and steady state.** Startup excluded, warm-up discarded, measurement in a declared
  steady-state window; cold start and compile / JIT time recorded as their own metrics.
- **M2 Controlled cache state.** Prefix / KV / compile caches reset or deliberately warmed, and recorded.
  A distinct seed per rep so reps cannot hit each other's prefix cache.
- **M3 Noise floor.** A/A runs (baseline vs itself) per worker type; a keep requires a gap beyond the
  measured noise (or a confidence interval over reps). Every comparison cites the floor it used.
- **M4 Interleaving.** Baseline and candidate run ABAB on the same worker to cancel thermal and
  background drift.
- **M5 Client is never the bottleneck.** The load generator runs on CPUs (or a host) separate from the
  engine; client CPU and event-loop lag are monitored; a run where the client saturated is invalid.
- **M6 Primary score: goodput at an SLO.** Sweep load (closed-loop) until TTFT p95 or TPOT p95 breaks
  the SLO of the workload profile; the score is output throughput at the last compliant load. A
  candidate must win on the minimum across the campaign's workloads.
- **M7 Distributions.** p50 / p95 / p99 and histograms, never a bare mean.
- **M8 Environment fingerprint.** Driver, firmware, clocks, power cap, ECC, co-tenant processes; warn on
  shared or throttling devices.

## 6. Quality gates

- **Q1 No-garbage fast fail.** Non-finite logprobs, repeated token 0, runs of `!`, replacement chars,
  request errors.
- **Q2 Token match vs reference.** Greedy, first N tokens, fixed quality set. The threshold is relative
  to the reference's own run-to-run match rate (some kernels flip near-tie tokens), not 100 %.
- **Q3 Drift guard.** Each new champion is also compared with the original baseline (looser threshold),
  so small drift cannot accumulate across keeps.
- **Q4 Robustness: tf-eval**, invoked automatically through the quality plugin; its score and run ID
  land in the arm's tracker run.

## 7. Metrics to watch

| Layer | Metrics | Why |
|---|---|---|
| Client SLIs | TTFT, TPOT / ITL, E2E (p50/p95/p99), req/s, output tok/s, goodput, errors, timeouts | the score; what users feel |
| Engine | running / waiting requests, KV-cache usage, prefix-cache hit rate, preemptions, batch size, prefill vs decode tokens per step, scheduler step time, spec-decode acceptance | explains why the score moved |
| Hardware | utilization, memory used / bandwidth, power, clocks, temperature, throttle events, interconnect | contention and throttling |
| Host | engine CPU per core, client CPU, memory, network | CPU-bound engine or client |
| Quality | garbage flags, token-match %, tf-eval score, logprob drift | hard gate |
| Lifecycle | cold start, compile / JIT time, cache hits, crash / OOM reason | iteration speed, operability |

Engine and hardware metrics are kept as time series for the whole run, so a tail-latency spike can be
lined up with, e.g., a preemption burst.

## 8. Profiling

- **P1** Profiled runs are separate from scored runs (profilers perturb).
- **P2** Same workload slice and a fixed window (N requests at concurrency C) so arms are comparable.
- **P3** Vendor-neutral output: kernel time grouped into families (attention, MoE, dense GEMM,
  quantization, norm / elementwise, communication, other), so profiles from different vendors line up.
- **P4** On demand in v1; rule-triggered (large win or loss) in v2.
- **P5** Host-side profiling (e.g. py-spy on scheduler / API server) alongside kernel profiling.

## 9. Fast iteration

- **F1 Fidelity tiers**, each gated by the previous, all with early fail:
  smoke (~1 min) → screen (5-10 min, short SLO sweep, one workload) → confirm (all workloads, reps) →
  quality (tf-eval) → replay (open-loop, full fidelity).
- **F2 Warm caches keyed by engine image digest** (compiled kernels, JIT, weights); config-only arms
  never rebuild.
- **F3 Early abort** on garbage, SLO break during screen, OOM, timeout.
- **F4 Resume and dry run.** Interrupted sweeps resume; dry run prints the resolved launch and protocol
  without touching hardware.
- **F5 Parallel workers** where the hardware allows, each with declared isolation.

## 10. Evidence: upstream and downstream

Every run is immutable and linked to its parent.

- **Upstream (inputs):** harness git SHA (+ diff if dirty); engine image digest; engine / torch /
  kernel-library / driver / firmware versions; model weights hash; fully resolved launch command and
  env; tuned-config hashes; workload hash and load-generator config; worker hardware fingerprint;
  parent arm and stated hypothesis.
- **Downstream (outputs):** raw per-request records (timestamps, token counts); summary metrics; engine
  and hardware time series; logs; profiler traces and family tables; quality outputs; decision
  (keep / discard / crash / invalid) with reason; forward links (PR, canary, tf-eval run).
- **Tracker requirements:** params, tags, step-wise metrics, artifacts, parent / child runs.
- **Reproduce:** "reproduce run R" re-executes from R's upstream record on matching hardware and must land
  within the noise floor.

## 11. Reference profile (user story)

> As an inference engineer I configure: engine = vLLM in Docker; load = Locust replay of our captured
> conversations (closed-loop for the SLO sweep, open-loop for the final replay); quality = token match
> as the gate plus tf-eval; tracker = MLflow. I start a campaign and get one MLflow parent run per
> campaign, one child run per arm, a frontier chart, and every kept arm reproducible from its run page.

## 12. v1 scope

In: the referee (sections 4-10), plugin interfaces, vLLM launcher, Locust / tf-eval / MLflow plugins,
a files-only tracker fallback, human-driven campaigns.

Out (v2+): autonomous agent loop and campaign `program.md`; further engines (SGLang, TRT-LLM); tuner
campaigns (aiter / CK / Triton config tuning) as first-class jobs; rule-triggered profiling; automatic
rollout (never automatic).

Search surface when the loop arrives: T1 config (flags, env) and T2 tuner-produced configs; source
patches and kernel authoring out of scope.

## 13. Acceptance criteria (v1)

1. One campaign config runs on two hardware vendors by swapping only the worker and profiler plugins.
2. The A/A noise floor is measured automatically and cited by every comparison.
3. Client-saturated runs are flagged invalid.
4. Any recorded arm reproduces from its tracker run alone, within the noise floor.
5. A tf-eval score lands in the arm's MLflow run without a manual step.
6. Profiles from different vendors appear in one kernel-family table.
7. Swapping MLflow for the files-only tracker needs no campaign-config change.

## 14. Lessons carried in from manual campaigns

Pitfalls seen in earlier hand-run engine tuning that these requirements exist to prevent:

- the load generator pinned to the same CPU slice as the engine (M5);
- reps regenerating identical prompts and hitting the prefix cache (M2);
- a flag passed twice being rejected by the engine, silently skipping a whole sweep (F3, dry run);
- compile / JIT caches copied while another run was writing them (F2, isolation);
- an accepted "improvement" smaller than run-to-run noise, as in autoresearch's seed change (M3);
- the evaluation depending on code the optimizer can edit (section 2, referee enforced).

## 15. Open items before design

- Entry points and output formats of the Locust replay and tf-eval plugins.
- SLO values (TTFT p95, TPOT p95) per workload profile.
- Tracker naming (MLflow experiment names) — approved by the owner before creation.
