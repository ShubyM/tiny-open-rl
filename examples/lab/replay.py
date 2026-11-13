"""Train on saved episodes again, to time the trainer alone at a given width. Every episode is used, with advantage = reward - the mean over all of them.

  GPUS=4 ROUNDS=3 LAB_ROOT=... python replay.py out/step-0.pkl [more.pkl ...]

The first round also checks that the trainer reads the episodes as the sampler wrote them: a new adapter changes nothing, so the trainer's logprob of each sampled token should be the one the sampler gave it.
"""

import asyncio
import json
import os
import pickle
import sys
import time

import lab_rl as h
import tinker
from tinker import types

ROUNDS = h.env("ROUNDS", 3)


async def main():
  episodes = [ep for path in sys.argv[1:] for ep in pickle.load(open(path, "rb")) if ep["sampled"]]
  mean = sum(ep["reward"] for ep in episodes) / len(episodes)
  data = [h.datum(ep, ep["reward"] - mean or 1.0) for ep in episodes]
  service = tinker.ServiceClient(base_url=h.BASE_URL, api_key=os.getenv("TINKER_API_KEY", "tml-tinyrl"))
  start = time.time()
  meta = {"openrl.gpus": str(h.GPUS)}
  trainer = await service.create_lora_training_client_async(h.MODEL, rank=h.RANK, seed=0, user_metadata=meta)
  print(json.dumps({"gpus": h.GPUS, "create_s": round(time.time() - start, 1), "model": trainer.model_id}), flush=True)
  for i in range(ROUNDS):
    start = time.time()
    forward = await (await trainer.forward_backward_async(data, "importance_sampling")).result_async()
    mid = time.time()
    await (await trainer.optim_step_async(types.AdamParams(learning_rate=h.LR))).result_async()
    line = {
      "gpus": h.GPUS,
      "round": i,
      "forward_backward_s": round(mid - start, 1),
      "optim_s": round(time.time() - mid, 2),
    }
    line |= {"loss": round(forward.metrics["loss:sum"], 2), "tokens": sum(d.model_input.length for d in data)}
    if i == 0:
      gaps = sorted(
        abs(out["logprobs"].data[at + k - 1] - lp)
        for ep, out in zip(episodes, forward.loss_fn_outputs)
        for at, _, logprobs in ep["sampled"]
        for k, lp in enumerate(logprobs)
      )
      line |= {
        "logprob_gap_mean": round(sum(gaps) / len(gaps), 4),
        "logprob_gap_p99": round(gaps[len(gaps) * 99 // 100], 3),
      }
      line |= {"logprob_gap_max": round(gaps[-1], 2), "compared": len(gaps)}
    print(json.dumps(line), flush=True)


asyncio.run(main())
