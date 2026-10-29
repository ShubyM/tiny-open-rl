import json
import random
import unittest

from tinyrl.schedule import (
  Pool,
  RunningPool,
  ShardLoad,
  choose_gpus,
  choose_pool,
  choose_sampler_pool,
  gpu_zones,
  place_pools,
  start_order,
)


class TestSchedule(unittest.TestCase):
  def test_pool_parse_and_pairs_with(self):
    p1 = Pool.parse("lora:base:0")
    self.assertEqual(p1.role, "trainer")
    self.assertTrue(p1.shared)
    self.assertEqual(p1.target, "base")

    p2 = Pool.parse("sampler:base")
    self.assertEqual(p2.role, "sampler")
    self.assertTrue(p2.shared)
    self.assertEqual(p2.target, "base")

    self.assertTrue(p1.pairs_with(p2))

    p3 = Pool.parse("run")
    self.assertEqual(p3.role, "trainer")
    self.assertFalse(p3.shared)
    self.assertEqual(p3.target, "run")

    p4 = Pool.parse("sampler:fft:run")
    self.assertEqual(p4.role, "sampler")
    self.assertFalse(p4.shared)
    self.assertEqual(p4.target, "run")

    self.assertTrue(p3.pairs_with(p4))
    self.assertFalse(p1.pairs_with(p3))

  def test_choose_pool(self):
    shards = [ShardLoad("lora:b:0", None, 0, 13)]
    d = choose_pool("r", "b", 1, False, shards, 8, target_util=0.7, new_run_util=0.04)
    self.assertEqual(d.choice, "lora:b:0")
    self.assertEqual(d.rule, "reuse_shard_under_target")

    shards[0] = ShardLoad("lora:b:0", None, 18, 18)
    d = choose_pool("r", "b", 1, False, shards, 8, target_util=0.7, new_run_util=0.04)
    self.assertEqual(d.choice, "lora:b:1")
    self.assertEqual(d.rule, "open_new_shard")

    shards = [ShardLoad("lora:b:0", 0.1, 5, 10)]
    d = choose_pool("r", "b", 1, False, shards, 8, target_util=0.7, new_run_util=0.04)
    self.assertEqual(d.choice, "lora:b:0")
    self.assertEqual(d.detail["per_run"], 0.02)

    d = choose_pool("r", "b", 0, False, shards, 8)
    self.assertEqual(d.choice, "r")
    self.assertEqual(d.rule, "dedicated_fft")

    d = choose_pool("r", "b", 1, True, shards, 8)
    self.assertEqual(d.choice, "r")
    self.assertEqual(d.rule, "dedicated_exclusive")

  def test_choose_sampler_pool(self):
    d = choose_sampler_pool("b", False, "r")
    self.assertEqual(d.choice, "sampler:fft:b")
    self.assertEqual(d.rule, "shared_fft_sampler")

    d = choose_sampler_pool("b", False, "r", True)
    self.assertEqual(d.choice, "sampler:fft:r")
    self.assertEqual(d.rule, "dedicated_fft_sampler")

    d = choose_sampler_pool("", False, "r")
    self.assertEqual(d.choice, "sampler:fft:r")
    self.assertEqual(d.rule, "dedicated_fft_sampler")

    d = choose_sampler_pool("b", True, "r", True)
    self.assertEqual(d.choice, "sampler:lora:r")
    self.assertEqual(d.rule, "dedicated_exclusive_sampler")

    d = choose_sampler_pool("b", True, "r", False)
    self.assertEqual(d.choice, "sampler:b")
    self.assertEqual(d.rule, "shared_lora_sampler")

  def test_place_pools(self):
    active = {"p1", "p2", "p3"}
    nodes = {"n1": 4, "n2": 4}
    placed = {"p1": "n1"}
    n, decs = place_pools(active, nodes, placed)
    self.assertEqual(n["p1"], "n1")
    self.assertEqual(n["p2"], "n2")
    self.assertEqual(n["p3"], "n1")
    self.assertEqual(decs["p2"].rule, "least_loaded_node")

    n, decs = place_pools({"p1"}, {"n2": 4}, {"p1": "n1"})
    self.assertEqual(n["p1"], "n2")
    self.assertEqual(decs["p1"].rule, "reassigned_from_dead_node")

  def test_start_order(self):
    pools = ["lora:b:0", "sampler:b", "run", "sampler:fft:run"]
    excl = {"run", "sampler:fft:run"}
    ordered = start_order(pools, excl)
    self.assertEqual(ordered, ["run", "sampler:fft:run", "lora:b:0", "sampler:b"])

  def test_8gpu_23job_adversarial_arrival_order(self):
    pools = [
      "lora:google/gemma-4-E2B:0",
      "lora:Qwen/Qwen3-0.6B-Base:0",
      "qwen-ts-1",
      "qwen-ts-2",
      "gemma-ts",
      "gemma-x",
      "sampler:google/gemma-4-E2B",
      "sampler:Qwen/Qwen3-0.6B-Base",
      "sampler:fft:Qwen/Qwen3-0.6B-Base",
      "sampler:fft:google/gemma-4-E2B",
      "sampler:fft:gemma-x",
    ]
    rnd = random.Random(42)

    for _ in range(1000):
      rnd.shuffle(pools)
      running = []
      excl = set()

      for p in pools:
        if p in ("gemma-x", "sampler:fft:gemma-x"):
          excl.add(p)
        dec = choose_gpus(p, 1, 8, running, excl)
        running.append(RunningPool(p, dec.choice))

      placements = {r.pool: r.gpus[0] for r in running}

      st_gpus = {placements["lora:google/gemma-4-E2B:0"], placements["lora:Qwen/Qwen3-0.6B-Base:0"]}
      self.assertEqual(st_gpus, {0, 1})

      self.assertEqual(placements["qwen-ts-1"], 2)
      self.assertEqual(placements["qwen-ts-2"], 2)
      self.assertEqual(placements["gemma-ts"], 2)

      self.assertEqual(placements["gemma-x"], 3)

      ss_gpus = {placements["sampler:google/gemma-4-E2B"], placements["sampler:Qwen/Qwen3-0.6B-Base"]}
      self.assertEqual(ss_gpus, {4, 5})

      self.assertEqual(placements["sampler:fft:Qwen/Qwen3-0.6B-Base"], 6)
      self.assertEqual(placements["sampler:fft:google/gemma-4-E2B"], 6)

      self.assertEqual(placements["sampler:fft:gemma-x"], 7)

  def test_2gpu_exclusive_node(self):
    excl = {"qwen-lora-x", "sampler:lora:qwen-lora-x"}
    running = []

    d1 = choose_gpus("qwen-lora-x", 1, 2, running, excl)
    running.append(RunningPool("qwen-lora-x", d1.choice))
    self.assertEqual(d1.choice, [0])

    d2 = choose_gpus("sampler:lora:qwen-lora-x", 1, 2, running, excl)
    running.append(RunningPool("sampler:lora:qwen-lora-x", d2.choice))
    self.assertEqual(d2.choice, [1])

  def test_fuzzer(self):
    rnd = random.Random(42)
    for _ in range(500):
      gpus = rnd.choice([1, 2, 4, 8, 16])
      width = rnd.choice([1, 2, 4])
      if width > gpus:
        continue

      active = set()
      excl = set()
      for _ in range(20):
        action = rnd.choice(["add_trainer", "add_sampler", "remove"])
        if action == "add_trainer":
          r = rnd.random()
          if r < 0.2:
            name = f"excl-{rnd.randint(1, 20)}"
            excl.add(name)
          elif r < 0.5:
            name = f"ts-{rnd.randint(1, 20)}"
          else:
            name = f"lora:b-{rnd.randint(1, 5)}:{rnd.randint(0, 2)}"
          active.add(name)
        elif action == "add_sampler":
          r = rnd.random()
          if excl and r < 0.25:
            name = f"sampler:fft:{rnd.choice(sorted(excl))}"
            excl.add(name)
          elif r < 0.55:
            name = f"sampler:fft:ts-{rnd.randint(1, 20)}"
          else:
            name = f"sampler:b-{rnd.randint(1, 5)}"
          active.add(name)
        elif action == "remove" and active:
          p = rnd.choice(list(active))
          active.remove(p)

      streaming_running = []
      streaming_excl = set()
      for p in list(active):
        if p in excl:
          streaming_excl.add(p)
        try:
          dec = choose_gpus(p, width, gpus, streaming_running, streaming_excl)
          streaming_running.append(RunningPool(p, dec.choice))
        except ValueError:
          pass

      ordered = start_order(active, excl)
      batched_running = []
      for p in ordered:
        try:
          dec = choose_gpus(p, width, gpus, batched_running, excl)
          json.dumps(dec.as_dict())
          self.assertTrue(dec.rule)
          self.assertTrue(dec.detail["candidates"])
          self.assertEqual(len(dec.choice), width)
          self.assertEqual(dec.choice, list(range(dec.choice[0], dec.choice[0] + width)))
          batched_running.append(RunningPool(p, dec.choice))
        except ValueError:
          pass

      for running, current_excl in [(batched_running, excl), (streaming_running, streaming_excl)]:
        parsed = [(r, Pool.parse(r.pool), set(r.gpus)) for r in running]
        trainers = {
          g for p, _, gs in parsed if p.pool not in current_excl and Pool.parse(p.pool).role == "trainer" for g in gs
        }
        samplers = {
          g for p, _, gs in parsed if p.pool not in current_excl and Pool.parse(p.pool).role == "sampler" for g in gs
        }

        if gpus >= 2 * width:
          for r1, p1, gs1 in parsed:
            for r2, p2, gs2 in parsed:
              if r1.pool != r2.pool and p1.pairs_with(p2):
                self.assertTrue(gs1.isdisjoint(gs2))

        z = gpu_zones(gpus, width)
        if z.partitioned and not current_excl:
          self.assertTrue(trainers.isdisjoint(samplers))
          for role, z_bounds in [("trainer", z.trainer), ("sampler", z.sampler)]:
            z0, z1 = z_bounds
            if (z1 - z0) >= 3 * width:
              split = z0 + max(width, ((2 * (z1 - z0)) // 3 // width) * width)
              shared_pools = {p.pool for p, pool, _ in parsed if pool.role == role and pool.shared}
              timesliced_pools = {
                p.pool for p, pool, _ in parsed if pool.role == role and not pool.shared and p.pool not in current_excl
              }
              if len(shared_pools) <= (split - z0) // width:
                shared_gpus = {g for p, pool, gs in parsed if p.pool in shared_pools for g in gs}
                timesliced_gpus = {g for p, pool, gs in parsed if p.pool in timesliced_pools for g in gs}
                self.assertTrue(shared_gpus.isdisjoint(timesliced_gpus))


if __name__ == "__main__":
  unittest.main()
