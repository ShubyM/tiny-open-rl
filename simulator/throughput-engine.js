/* A seeded, discrete-event thought experiment, not a hardware benchmark. */
(function (root) {
  'use strict';

  const defaults = Object.freeze({
    gpus: 4, arrivals: 6, steps: 40, tokens: 4096, referenceTokens: 4096,
    stepSeconds: 0.6, switchSeconds: 0.5, switchMB: 0, quantum: 4,
    controlMs: 2, bytesPerToken: 64, bandwidthMB: 200,
    arrivalCV: 1, workCV: 0.5, serviceCV: 0.2, pauseSeconds: 0,
    duration: 600, warmup: 120, seed: 7, eventCap: 250000, sampleCount: 121
  });
  const limits = {
    gpus: [1, 256, true], arrivals: [0, 1000000], steps: [1, 1000000],
    tokens: [1, 16777216], referenceTokens: [1, 16777216],
    stepSeconds: [0.000001, 3600], switchSeconds: [0, 3600], switchMB: [0, 1048576], quantum: [1, 1000000, true],
    controlMs: [0, 60000], bytesPerToken: [0, 1000000], bandwidthMB: [0.001, 1000000],
    arrivalCV: [0, 10], workCV: [0, 5], serviceCV: [0, 5], pauseSeconds: [0, 86400],
    duration: [0.01, 86400], warmup: [0, 86400], seed: [0, 4294967295, true],
    eventCap: [1, 250000, true], sampleCount: [2, 1001, true]
  };
  const PHASES = ['compute', 'switch', 'control', 'data', 'idle'];
  const TRACE_LIMIT = 20000;

  function validateInputs(overrides) {
    const p = { ...defaults };
    for (const name of Object.keys(limits)) {
      if (overrides && Object.prototype.hasOwnProperty.call(overrides, name)) p[name] = overrides[name];
      const [lo, hi, integer] = limits[name];
      if (typeof p[name] !== 'number' || !Number.isFinite(p[name]) || p[name] < lo || p[name] > hi ||
          (integer && !Number.isInteger(p[name]))) {
        throw new RangeError(`${name} must be ${integer ? 'an integer' : 'a finite number'} in [${lo}, ${hi}]`);
      }
    }
    if (p.warmup >= p.duration) throw new RangeError('warmup must be less than duration');
    return p;
  }

  function random(seed) {
    let state = seed >>> 0;
    let spare = null;
    const uniform = () => {
      state = (state + 0x6D2B79F5) >>> 0;
      let t = Math.imul(state ^ (state >>> 15), state | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const positive = () => Math.max(1 / 4294967296, uniform());
    const normal = () => {
      if (spare !== null) { const value = spare; spare = null; return value; }
      const radius = Math.sqrt(-2 * Math.log(positive()));
      const angle = 2 * Math.PI * uniform();
      spare = radius * Math.sin(angle);
      return radius * Math.cos(angle);
    };
    const gamma = shape => {
      if (shape < 1) return gamma(shape + 1) * Math.pow(positive(), 1 / shape);
      const d = shape - 1 / 3;
      const c = 1 / Math.sqrt(9 * d);
      for (;;) {
        const x = normal();
        const v0 = 1 + c * x;
        if (v0 <= 0) continue;
        const v = v0 * v0 * v0;
        const u = positive();
        if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < x * x / 2 + d * (1 - v + Math.log(v))) return d * v;
      }
    };
    const lognormal = (mean, cv) => {
      if (cv === 0) return mean;
      const variance = Math.log1p(cv * cv);
      return mean * Math.exp(Math.sqrt(variance) * normal() - variance / 2);
    };
    return { uniform, positive, normal, gamma, lognormal };
  }

  class MinHeap {
    constructor() { this.items = []; }
    static before(a, b) { return a.t < b.t || (a.t === b.t && a.id < b.id); }
    push(value) {
      const a = this.items;
      let i = a.length;
      a.push(value);
      while (i > 0) {
        const parent = (i - 1) >> 1;
        if (!MinHeap.before(value, a[parent])) break;
        a[i] = a[parent]; i = parent;
      }
      a[i] = value;
    }
    pop() {
      const a = this.items;
      const first = a[0], last = a.pop();
      if (a.length) {
        let i = 0;
        while (2 * i + 1 < a.length) {
          let child = 2 * i + 1;
          if (child + 1 < a.length && MinHeap.before(a[child + 1], a[child])) child++;
          if (!MinHeap.before(a[child], last)) break;
          a[i] = a[child]; i = child;
        }
        a[i] = last;
      }
      return first;
    }
    peek() { return this.items[0]; }
  }

  class FIFO {
    constructor() { this.items = []; this.head = 0; }
    push(item) { this.items.push(item); }
    shift() {
      const item = this.items[this.head++];
      if (this.head > 1024 && this.head * 2 > this.items.length) {
        this.items = this.items.slice(this.head); this.head = 0;
      }
      return item;
    }
    get length() { return this.items.length - this.head; }
  }

  function quantile(values, fraction) {
    if (!values.length) return null;
    const a = values.slice().sort((x, y) => x - y);
    const position = (a.length - 1) * fraction;
    const i = Math.floor(position);
    return a[i] + (a[Math.min(i + 1, a.length - 1)] - a[i]) * (position - i);
  }

  function capacityBounds(p) {
    const computeSeconds = p.stepSeconds * p.tokens / p.referenceTokens;
    const controlSeconds = p.controlMs / 1000;
    const dataSeconds = p.tokens * p.bytesPerToken / (p.bandwidthMB * 1048576);
    const compute = p.gpus / computeSeconds;
    const control = controlSeconds ? 1 / controlSeconds : null;
    const data = dataSeconds ? 1 / dataSeconds : null;
    const switched = p.gpus / (computeSeconds + p.switchSeconds / p.quantum);
    const compound = p.gpus / (computeSeconds + p.switchSeconds / p.quantum + controlSeconds + dataSeconds);
    const bottleneck = Math.min(compute, control ?? Infinity, data ?? Infinity);
    return {
      meanComputeSeconds: computeSeconds, meanControlSeconds: controlSeconds, meanDataSeconds: dataSeconds,
      stateTransferSecondsPerInstall: p.switchMB / p.bandwidthMB,
      computeStepsPerSec: compute, switchBudgetStepsPerSec: switched,
      controlStepsPerSec: control, dataStepsPerSec: data, compoundSlotStepsPerSec: compound,
      resourceUpperStepsPerSec: bottleneck, resourceUpperTokensPerSec: bottleneck * p.tokens,
      offeredStepsPerSec: p.arrivals * p.steps / 60, offeredTokensPerSec: p.arrivals * p.steps * p.tokens / 60,
      assumptions: 'Compute/control/data are isolated-resource ceilings. Switch-budget and compound-slot rates assume '
        + 'a changed job every full quantum; affinity, short jobs, finite load and queueing change the observed rate. '
        + 'These simple rates omit optional state-installation traffic, which competes on the same data link. '
        + 'Local switch time excludes that separately configured state transfer. '
        + 'Token-time scaling is a local linear approximation; these are not hardware measurements.'
    };
  }

  function simulate(overrides) {
    const p = validateInputs(overrides);
    const heap = new MinHeap();
    const arrivalsRng = random(p.seed ^ 0xA341316C);
    const workRng = random(p.seed ^ 0xC8013EA4);
    const ready = new FIFO();
    const jobs = [];
    const samples = [];
    let trace = [], traceHead = 0, intervalsTruncated = false;
    let now = 0, eventId = 0, eventsProcessed = 0;
    let cumulativeTokens = 0, cumulativeSteps = 0, cumulativeCompleted = 0, cumulativeArrivedSteps = 0;
    let pendingSteps = 0, windowTokens = 0, windowSteps = 0, switchCount = 0;
    let completedTokenSum = 0, completedComputeSum = 0;
    const workers = Array.from({ length: p.gpus }, (_, id) => ({
      id, job: null, lastJob: null, phase: 'idle', phaseJob: null, since: 0,
      totals: Object.fromEntries(PHASES.map(phase => [phase, 0]))
    }));
    const inWindow = time => time >= p.warmup;
    const overlap = (start, end) => Math.max(0, end - Math.max(start, p.warmup));
    const schedule = (time, fn) => {
      if (!Number.isFinite(time)) throw new RangeError('non-finite event time');
      heap.push({ t: Math.max(time, now), id: eventId++, fn });
    };

    function recordInterval(worker, end) {
      worker.totals[worker.phase] += overlap(worker.since, end);
      if (worker.id < 8 && end > worker.since) {
        trace.push({ gpu: worker.id, phase: worker.phase, start: worker.since, end, job: worker.phaseJob });
        while (traceHead < trace.length && trace[traceHead].end < end - 60) traceHead++;
        while (trace.length - traceHead > TRACE_LIMIT) { traceHead++; intervalsTruncated = true; }
        if (traceHead > 2048 && traceHead * 2 > trace.length) { trace = trace.slice(traceHead); traceHead = 0; }
      }
    }
    function phase(worker, name, job) {
      const jobId = job ? job.id : null;
      if (worker.phase === name && worker.phaseJob === jobId) return;
      recordInterval(worker, now);
      worker.phase = name; worker.phaseJob = jobId; worker.since = now;
    }

    class Server {
      constructor(name) { this.name = name; this.queue = new FIFO(); this.current = null; this.busySeconds = 0; }
      submit(worker, duration, done) {
        phase(worker, this.name, worker.job);
        // An empty step payload must not queue behind state installation.
        // Keep the original event path when optional state traffic is disabled.
        if (this.name === 'data' && duration === 0 && p.switchMB > 0) {
          schedule(now, done);
          return;
        }
        const request = { duration, done };
        if (this.current) this.queue.push(request);
        else this.begin(request);
      }
      begin(request) {
        this.current = { ...request, start: now };
        schedule(now + request.duration, () => {
          const active = this.current;
          this.busySeconds += overlap(active.start, now);
          this.current = null;
          // Preserve FIFO: older waiters start before the completed request can
          // enqueue its next phase or another step.
          if (this.queue.length) this.begin(this.queue.shift());
          active.done();
        });
      }
      finishWindow(end) {
        if (this.current) this.busySeconds += overlap(this.current.start, end);
      }
    }
    const control = new Server('control');
    const transfer = new Server('data');

    function dispatch() {
      while (ready.length) {
        const idle = workers.filter(worker => worker.job === null);
        if (!idle.length) return;
        const job = ready.shift();
        const worker = idle.find(candidate => candidate.lastJob === job.id) || idle[0];
        worker.job = job;
        job.visitSteps = 0;
        if (worker.lastJob !== job.id) {
          if (inWindow(now)) switchCount++;
          worker.lastJob = job.id;
          const localSwitch = () => {
            phase(worker, 'switch', job);
            schedule(now + p.switchSeconds * job.switchRng.lognormal(1, p.serviceCV), () => step(worker));
          };
          if (p.switchMB > 0) transfer.submit(worker, p.switchMB / p.bandwidthMB, localSwitch);
          else localSwitch();
        } else step(worker);
      }
    }

    function step(worker) {
      const job = worker.job;
      const tokens = Math.max(1, Math.round(job.serviceRng.lognormal(p.tokens, p.serviceCV)));
      const computeSeconds = p.stepSeconds * tokens / p.referenceTokens * job.serviceRng.lognormal(1, p.serviceCV);
      const dataSeconds = tokens * p.bytesPerToken / (p.bandwidthMB * 1048576);
      control.submit(worker, p.controlMs / 2000, () => {
        transfer.submit(worker, dataSeconds * 0.8, () => {
          phase(worker, 'compute', job);
          if (job.firstStart === null) job.firstStart = now;
          schedule(now + computeSeconds, () => {
            transfer.submit(worker, dataSeconds * 0.2, () => {
              control.submit(worker, p.controlMs / 2000, () => {
                cumulativeTokens += tokens; cumulativeSteps++; pendingSteps--;
                if (inWindow(now)) {
                  windowTokens += tokens; windowSteps++;
                  completedTokenSum += tokens; completedComputeSum += computeSeconds;
                }
                job.remaining--; job.visitSteps++;
                if (job.remaining === 0) {
                  job.completed = now; cumulativeCompleted++;
                  release(worker);
                } else if (job.visitSteps >= p.quantum) {
                  release(worker, true);
                } else step(worker);
              });
            });
          });
        });
      });
    }

    function release(worker, requeue) {
      const job = worker.job;
      worker.job = null;
      phase(worker, 'idle', null);
      if (requeue) {
        if (p.pauseSeconds > 0) schedule(now + p.pauseSeconds, () => { ready.push(job); dispatch(); });
        else ready.push(job);
      }
      dispatch();
    }

    function interarrival() {
      const mean = 60 / p.arrivals;
      if (p.arrivalCV === 0) return mean;
      if (p.arrivalCV === 1) return -Math.log(arrivalsRng.positive()) * mean;
      const shape = 1 / (p.arrivalCV * p.arrivalCV);
      return Math.max(1e-9, arrivalsRng.gamma(shape) * mean / shape);
    }
    function arrive() {
      const raw = workRng.lognormal(p.steps, p.workCV);
      // Stochastic rounding preserves fractional means before the necessary
      // minimum of one step. The observed mean is reported rather than hidden.
      const steps = Math.max(1, Math.floor(raw) + (workRng.uniform() < raw % 1 ? 1 : 0));
      const id = jobs.length;
      const job = {
        id, arrival: now, steps, remaining: steps, firstStart: null, completed: null, visitSteps: 0,
        serviceRng: random(p.seed ^ Math.imul(id + 1, 0x9E3779B1) ^ 0xAD90777D),
        switchRng: random(p.seed ^ Math.imul(id + 1, 0x85EBCA77) ^ 0x7E95761E)
      };
      jobs.push(job); ready.push(job); cumulativeArrivedSteps += steps; pendingSteps += steps;
      dispatch();
      const next = now + interarrival();
      if (next <= p.duration) schedule(next, arrive);
    }
    function sample() {
      const value = {
        t: now, ready: ready.length, unfinished: jobs.length - cumulativeCompleted, pendingSteps,
        cumulativeTokens, cumulativeSteps, cumulativeCompleted, cumulativeArrivedJobs: jobs.length,
        cumulativeArrivedSteps, controlQueue: control.queue.length, transferQueue: transfer.queue.length
      };
      if (samples.length && samples[samples.length - 1].t === now) samples[samples.length - 1] = value;
      else samples.push(value);
    }

    if (p.arrivals > 0) {
      const first = interarrival();
      if (first <= p.duration) schedule(first, arrive);
    }
    for (let i = 0; i < p.sampleCount; i++) schedule(p.duration * i / (p.sampleCount - 1), sample);
    let truncated = false;
    while (heap.peek() && heap.peek().t <= p.duration) {
      if (eventsProcessed >= p.eventCap) { truncated = true; break; }
      const event = heap.pop(); now = event.t; event.fn(); eventsProcessed++;
    }
    const end = truncated ? now : p.duration;
    now = end;
    sample();
    for (const worker of workers) recordInterval(worker, end);
    control.finishWindow(end); transfer.finishWindow(end);
    const measurementStart = Math.min(p.warmup, end);
    const seconds = Math.max(0, end - p.warmup);
    const arrivals = jobs.filter(job => job.arrival >= p.warmup);
    const completed = jobs.filter(job => job.completed !== null && job.completed >= p.warmup);
    const started = jobs.filter(job => job.firstStart !== null && job.arrival >= p.warmup);
    const firstWaits = started.map(job => job.firstStart - job.arrival);
    const latencies = completed.map(job => job.completed - job.arrival);
    const unfinished = jobs.length - cumulativeCompleted;
    const beforeWindow = jobs.filter(job => job.arrival < p.warmup &&
      (job.completed === null || job.completed >= p.warmup)).length;
    const gpuFractions = Object.fromEntries(PHASES.map(name => [name, seconds ?
      workers.reduce((sum, worker) => sum + worker.totals[name], 0) / (seconds * p.gpus) : (name === 'idle' ? 1 : 0)]));
    const cutoff = Math.max(0, end - 60);
    const intervals = trace.slice(traceHead).filter(interval => interval.end > cutoff)
      .map(interval => ({ ...interval, start: Math.max(interval.start, cutoff) }));
    const summary = {
      measuredSeconds: seconds, tokensPerSec: seconds ? windowTokens / seconds : 0,
      jobsPerMin: seconds ? completed.length * 60 / seconds : 0,
      completed: completed.length, arrived: arrivals.length, unfinished, completedSteps: windowSteps,
      backlogGrowthJobsPerMin: seconds ? (unfinished - beforeWindow) * 60 / seconds : 0,
      p50FirstStartWait: quantile(firstWaits, 0.5), p95FirstStartWait: quantile(firstWaits, 0.95),
      p50CompletedLatency: quantile(latencies, 0.5), p95CompletedLatency: quantile(latencies, 0.95),
      startedObservations: started.length, completedLatencyObservations: completed.length,
      unstartedWindowArrivals: arrivals.filter(job => job.firstStart === null).length,
      unfinishedWindowArrivals: arrivals.filter(job => job.completed === null).length,
      remainingSteps: pendingSteps, workRemainingSteps: pendingSteps,
      gpuFractions, controlUtil: seconds ? control.busySeconds / seconds : 0,
      transferUtil: seconds ? transfer.busySeconds / seconds : 0,
      switchRate: seconds ? switchCount / seconds : 0, switchCount,
      meanJobSteps: jobs.length ? cumulativeArrivedSteps / jobs.length : null,
      meanTokensPerCompletedStep: windowSteps ? completedTokenSum / windowSteps : null,
      meanComputeSecondsPerCompletedStep: windowSteps ? completedComputeSum / windowSteps : null,
      cumulativeTokens, cumulativeSteps, cumulativeCompleted, cumulativeArrivedJobs: jobs.length,
      latencyCaveat: 'First-start wait is arrival to first compute, for window arrivals that started. '
        + 'Job latency includes only jobs completed in the window. Unstarted/unfinished jobs are censored, not zero-latency.'
    };
    return {
      inputs: p, summary, samples, intervals, capacities: capacityBounds(p),
      truncated, truncationReason: truncated ? 'event-cap' : null, eventsProcessed,
      measurementStart, measurementEnd: end, intervalsTruncated,
      traceStart: cutoff, traceEnd: end, traceGPUs: Math.min(8, p.gpus)
    };
  }

  const api = { simulate, defaults, validateInputs, capacityBounds };
  root.RLSim = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
