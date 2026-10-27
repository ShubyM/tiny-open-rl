"""The Tinker SDK, unchanged, training a LoRA through tiny-open-rl. Run it with the SDK installed:

BASE_URL=http://localhost:8000 BASE_MODEL=Qwen/Qwen3-0.6B python examples/tinker_sdk.py
"""

import os

import tinker
from tinker import types

client = tinker.ServiceClient(
  api_key=os.getenv("TINKER_API_KEY", "tml-tinyrl"), base_url=os.getenv("BASE_URL", "http://localhost:8000")
)
trainer = client.create_lora_training_client(base_model=os.environ["BASE_MODEL"], rank=8, seed=0)
print(f"model {trainer.model_id}")

# learn to count: each token's target is the next one
tokens = list(range(1, 33))
datum = types.Datum(
  model_input=types.ModelInput.from_ints(tokens[:-1]),
  loss_fn_inputs={"target_tokens": tokens[1:], "weights": [1.0] * (len(tokens) - 1)},
)

losses = []
for step in range(int(os.getenv("STEPS", "10"))):
  fwdbwd = trainer.forward_backward([datum], "cross_entropy")
  optim = trainer.optim_step(types.AdamParams(learning_rate=float(os.getenv("LR", "1e-2"))))
  losses.append(fwdbwd.result().metrics["loss:sum"])
  print(f"step {step:2d}  loss {losses[-1]:8.4f}  grad norm {optim.result().metrics['grad_norm:mean']:.4f}")

state = trainer.save_state("final").result().path
trainer.load_state_with_optimizer(state).result()
sampler = trainer.save_weights_for_sampler("final").result().path
print(f"state   {state}")
print(f"sampler {sampler}")

assert losses[-1] < losses[0] * 0.7, f"the loss barely moved: {losses}"

sampler_client = trainer.save_weights_and_get_sampling_client()
print(f"sampler_session {sampler_client._sampling_session_id}")

prompt_tokens = [1, 2, 3]
out = sampler_client.sample(
  prompt=types.ModelInput.from_ints(prompt_tokens),
  num_samples=2,
  sampling_params=types.SamplingParams(max_tokens=4, temperature=1.0),
).result()

seq = out.sequences[0]
print(f"sampled {seq.tokens} stop_reason {seq.stop_reason}")

prompt_lps = sampler_client.compute_logprobs(
  types.ModelInput.from_ints(prompt_tokens + list(seq.tokens)), prompt_logprobs_last_n=len(seq.tokens)
).result()

is_datum = types.Datum(
  model_input=types.ModelInput.from_ints(prompt_tokens + list(seq.tokens)[:-1]),
  loss_fn_inputs={
    "target_tokens": list(seq.tokens),
    "logprobs": list(seq.logprobs),
    "advantages": [1.0] * len(seq.tokens),
    "weights": [1.0] * len(seq.tokens),
  },
)
fwdbwd_is = trainer.forward_backward([is_datum], "importance_sampling")
optim_is = trainer.optim_step(types.AdamParams(learning_rate=1e-3))
print(f"importance_sampling loss {fwdbwd_is.result().metrics['loss:sum']:.4f}")
