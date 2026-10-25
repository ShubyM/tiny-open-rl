'use strict';
const assert = require('node:assert/strict');
const { simulate, defaults, validateInputs } = require('./throughput-engine.js');
let passed = 0;
function test(name, fn) { fn(); passed++; console.log(`ok ${passed} - ${name}`); }
function near(actual, expected, tolerance = 1e-9) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected} by more than ${tolerance}`);
}
const deterministic = {
  gpus: 4, arrivals: 600, steps: 100, tokens: 4096, referenceTokens: 4096,
  stepSeconds: 0.1, switchSeconds: 0, quantum: 4, controlMs: 0, bytesPerToken: 0,
  arrivalCV: 0, workCV: 0, serviceCV: 0, pauseSeconds: 0, duration: 120, warmup: 20
};
function conservation(result) {
  const s = result.summary;
  near(Object.values(s.gpuFractions).reduce((a, b) => a + b, 0), 1, 1e-9);
  for (const fraction of Object.values(s.gpuFractions)) assert.ok(fraction >= -1e-10 && fraction <= 1 + 1e-10);
  assert.ok(s.controlUtil >= -1e-10 && s.controlUtil <= 1 + 1e-10);
  assert.ok(s.transferUtil >= -1e-10 && s.transferUtil <= 1 + 1e-10);
  if (s.measuredSeconds) near(s.backlogGrowthJobsPerMin, (s.arrived - s.completed) * 60 / s.measuredSeconds);
  for (const sample of result.samples) {
    assert.equal(sample.pendingSteps, sample.cumulativeArrivedSteps - sample.cumulativeSteps);
    assert.equal(sample.unfinished, sample.cumulativeArrivedJobs - sample.cumulativeCompleted);
    assert.ok(sample.ready >= 0 && sample.ready <= sample.unfinished);
    assert.ok(sample.controlQueue >= 0 && sample.transferQueue >= 0);
  }
  const end = result.samples.at(-1);
  assert.equal(end.pendingSteps, s.remainingSteps);
  assert.equal(end.unfinished, s.unfinished);
  assert.equal(end.cumulativeTokens, s.cumulativeTokens);
  for (let gpu = 0; gpu < result.traceGPUs; gpu++) {
    const intervals = result.intervals.filter(item => item.gpu === gpu);
    for (let i = 0; i < intervals.length; i++) {
      assert.ok(intervals[i].end > intervals[i].start);
      if (i) assert.ok(intervals[i].start >= intervals[i - 1].end - 1e-9, 'one GPU cannot overlap phases');
    }
  }
}

test('deterministic saturated zero-overhead throughput reaches the compute ceiling', () => {
  const r = simulate(deterministic);
  near(r.summary.completedSteps / r.summary.measuredSeconds, 4 / 0.1, 0.05);
  near(r.summary.tokensPerSec, 4 / 0.1 * 4096, 0.05 * 4096);
  near(r.summary.gpuFractions.compute, 1, 1e-9);
  conservation(r);
});

test('overload remains unfinished at the horizon instead of draining invisibly', () => {
  const r = simulate(deterministic);
  assert.equal(r.measurementEnd, deterministic.duration);
  assert.ok(r.summary.unfinished > 1000);
  assert.ok(r.summary.backlogGrowthJobsPerMin > 500);
  assert.ok(r.samples.at(-1).pendingSteps > r.samples[60].pendingSteps);
  conservation(r);
});

test('larger quanta amortize switches while delaying first service for competitors', () => {
  const p = { ...deterministic, gpus: 1, arrivals: 60, steps: 40, stepSeconds: 0.05,
    switchSeconds: 0.2, duration: 200, warmup: 20 };
  const small = simulate({ ...p, quantum: 1 });
  const large = simulate({ ...p, quantum: 20 });
  assert.ok(large.summary.tokensPerSec > small.summary.tokensPerSec * 3);
  assert.ok(large.summary.switchRate < small.summary.switchRate / 3);
  assert.ok(large.summary.p95FirstStartWait > small.summary.p95FirstStartWait);
  assert.equal(small.summary.cumulativeArrivedJobs, large.summary.cumulativeArrivedJobs);
  assert.equal(small.samples.at(-1).cumulativeArrivedSteps, large.samples.at(-1).cumulativeArrivedSteps);
  conservation(small); conservation(large);
});

test('serialized control limits aggregate throughput even with four GPUs', () => {
  const r = simulate({ ...deterministic, stepSeconds: 0.01, controlMs: 100 });
  near(r.summary.completedSteps / r.summary.measuredSeconds, 10, 0.03);
  near(r.summary.controlUtil, 1, 1e-9);
  assert.ok(r.summary.gpuFractions.control > 0.95);
  assert.ok(r.summary.gpuFractions.compute < 0.04);
  conservation(r);
});

test('the single FIFO data link limits throughput and blocks occupied GPU slots', () => {
  const r = simulate({ ...deterministic, stepSeconds: 0.01, bytesPerToken: 256, bandwidthMB: 1 });
  // Each 4096-token step transfers exactly 1 MiB in total.
  near(r.summary.completedSteps / r.summary.measuredSeconds, 1, 0.03);
  near(r.summary.transferUtil, 1, 1e-9);
  assert.ok(r.summary.gpuFractions.data > 0.99);
  conservation(r);
});

test('optional state installation shares the link, including the first job visit', () => {
  assert.deepEqual(simulate(deterministic), simulate({ ...deterministic, switchMB: 0 }));
  const r = simulate({ ...deterministic, steps: 1, stepSeconds: 0.01,
    switchMB: 1, bandwidthMB: 1 });
  // Each new one-step job installs 1 MiB, with no per-token data traffic.
  near(r.summary.completedSteps / r.summary.measuredSeconds, 1, 0.03);
  near(r.summary.transferUtil, 1, 1e-9);
  assert.ok(r.summary.gpuFractions.data > 0.99);
  near(r.capacities.stateTransferSecondsPerInstall, 1);
  conservation(r);
  const cold = simulate({ ...deterministic, gpus: 1, arrivals: 1, steps: 4,
    stepSeconds: 1, switchSeconds: 0.5, switchMB: 2, bandwidthMB: 1,
    quantum: 1, duration: 69.5, warmup: 60 });
  assert.equal(cold.summary.switchCount, 1, 'same-job visits reuse the installed state');
  near(cold.summary.p50FirstStartWait, 2.5, 1e-9);
  assert.equal(cold.summary.completedSteps, 4);
  conservation(cold);
});

test('reference-token calibration makes token increases cost proportional GPU time', () => {
  const one = simulate({ ...deterministic, tokens: 4096 });
  const two = simulate({ ...deterministic, tokens: 8192 });
  near(one.capacities.meanComputeSeconds, 0.1);
  near(two.capacities.meanComputeSeconds, 0.2);
  // Window boundaries can include different partial sets of four parallel steps.
  near(one.summary.tokensPerSec, two.summary.tokensPerSec, 4 * (4096 + 8192) / 100);
  near(one.summary.completedSteps / two.summary.completedSteps, 2, 0.01);
});

test('cold installation is charged and client pause releases the GPU', () => {
  const r = simulate({ ...deterministic, gpus: 1, arrivals: 1, steps: 4,
    stepSeconds: 1, switchSeconds: 0.5, quantum: 1, pauseSeconds: 3, duration: 69.5, warmup: 60 });
  assert.equal(r.summary.switchCount, 1, 'same-job affinity should avoid repeat installations');
  assert.equal(r.summary.completedSteps, 3);
  near(r.summary.p50FirstStartWait, 0.5);
  assert.ok(r.summary.gpuFractions.idle > 0.6);
  assert.equal(r.summary.unfinished, 1);
  conservation(r);
});

test('seeds reproduce all results and nonconstant random workloads preserve conservation', () => {
  const first = simulate();
  assert.deepEqual(first, simulate());
  assert.notDeepEqual(first.summary, simulate({ seed: defaults.seed + 1 }).summary);
  conservation(first);
  for (const arrivalCV of [0.2, 1, 2.5]) {
    conservation(simulate({ arrivalCV, workCV: 1, serviceCV: 0.7, gpus: 3, pauseSeconds: 0.4 }));
  }
});

test('unfinished jobs are censored rather than counted as fast completions', () => {
  const r = simulate({ ...deterministic, gpus: 1, arrivals: 60, steps: 1000,
    stepSeconds: 1, quantum: 1, duration: 20, warmup: 0 });
  assert.equal(r.summary.completed, 0);
  assert.equal(r.summary.p50CompletedLatency, null);
  assert.equal(r.summary.p95CompletedLatency, null);
  assert.equal(r.summary.completedLatencyObservations, 0);
  assert.ok(r.summary.unstartedWindowArrivals > 0);
  assert.ok(r.summary.unfinishedWindowArrivals > 0);
  assert.ok(r.summary.startedObservations > 0);
  conservation(r);
});

test('no arrivals produce an idle system with no invented latency observations', () => {
  const r = simulate({ arrivals: 0, duration: 10, warmup: 0 });
  assert.equal(r.summary.tokensPerSec, 0);
  assert.equal(r.summary.unfinished, 0);
  assert.equal(r.summary.p95FirstStartWait, null);
  assert.equal(r.summary.gpuFractions.idle, 1);
  assert.equal(r.samples.length, 121);
  conservation(r);
});

test('event-cap truncation is explicit and stops the measured window at simulated time', () => {
  const r = simulate({ ...deterministic, arrivals: 100000, duration: 600, warmup: 0, eventCap: 100 });
  assert.equal(r.truncated, true);
  assert.equal(r.truncationReason, 'event-cap');
  assert.equal(r.eventsProcessed, 100);
  assert.ok(r.measurementEnd < 600);
  assert.equal(r.samples.at(-1).t, r.measurementEnd);
  near(r.summary.measuredSeconds, r.measurementEnd);
  conservation(r);
});

test('invalid inputs fail clearly and user cannot bypass the finite event cap', () => {
  for (const bad of [{ gpus: 0 }, { arrivals: -1 }, { stepSeconds: 0 }, { bandwidthMB: 0 },
    { quantum: 1.5 }, { seed: NaN }, { eventCap: 250001 }, { warmup: 600 }, { referenceTokens: 0 }, { switchMB: -1 }]) {
    assert.throws(() => validateInputs(bad), RangeError);
  }
});

console.log(`${passed} simulator tests passed`);
