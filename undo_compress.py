with open("tinyrl/workers/model.py") as f:
  text = f.read()

text = text.replace(
  """    except Exception:
      if not forward_only and total > 0: self.needs_recovery.add(run)
      raise""",
  """    except Exception:
      if not forward_only and total > 0:
        self.needs_recovery.add(run)
      raise""",
)

text = text.replace(
  """    for name, p in self.params[run].items():
      if weights[name].shape != p.shape: raise ValueError(f"shape mismatch for {name}")""",
  """    for name, p in self.params[run].items():
      if weights[name].shape != p.shape:
        raise ValueError(f"shape mismatch for {name}")""",
)

text = text.replace(
  """    if optimizer and any(k.endswith(".step") for k in tensors):
      for name in self.params[run]:
        if not all(f"{name}.{s}" in tensors for s in ("exp_avg", "exp_avg_sq", "step")): raise ValueError("missing opt")""",
  """    if optimizer and any(k.endswith(".step") for k in tensors):
      for name in self.params[run]:
        if not all(f"{name}.{s}" in tensors for s in ("exp_avg", "exp_avg_sq", "step")):
          raise ValueError("missing opt")""",
)

text = text.replace(
  """  def optim_step(self, run: str, adam: dict) -> dict[str, float]:
    if run in getattr(self, "needs_recovery", set()): raise RuntimeError("run needs recovery")""",
  """  def optim_step(self, run: str, adam: dict) -> dict[str, float]:
    if run in getattr(self, "needs_recovery", set()):
      raise RuntimeError("run needs recovery")""",
)

with open("tinyrl/workers/model.py", "w") as f:
  f.write(text)

with open("tinyrl/workers/trainer.py") as f:
  text = f.read()

text = text.replace(
  """    if not reply.ok and created_here: await asyncio.to_thread(self.model.delete_run, cmd.run)""",
  """    if not reply.ok and created_here:
      await asyncio.to_thread(self.model.delete_run, cmd.run)""",
)

with open("tinyrl/workers/trainer.py", "w") as f:
  f.write(text)
