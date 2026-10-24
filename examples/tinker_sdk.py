# the tinker sdk, unchanged, training a lora through tiny-open-rl. run it with the sdk installed:
#   BASE_URL=http://localhost:8000 BASE_MODEL=Qwen/Qwen3-0.6B python examples/tinker_sdk.py
import os
import tinker
from tinker import types

client = tinker.ServiceClient(api_key=os.getenv("TINKER_API_KEY", "tml-tinyrl"), base_url=os.getenv("BASE_URL", "http://localhost:8000"))
trainer = client.create_lora_training_client(base_model=os.environ["BASE_MODEL"], rank=8, seed=0)
print(f"model {trainer.model_id}")

# learn to count: every token's target is the next one
tokens = list(range(1, 33))
inputs = {"target_tokens": tokens[1:], "weights": [1.0]*(len(tokens)-1)}
datum = types.Datum(model_input=types.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs=inputs)

losses = []
for step in range(int(os.getenv("STEPS", "10"))):
  fwdbwd = trainer.forward_backward([datum], "cross_entropy")
  optim = trainer.optim_step(types.AdamParams(learning_rate=float(os.getenv("LR", "1e-2"))))
  losses.append(fwdbwd.result().metrics["loss:sum"])
  print(f"step {step:2d}  loss {losses[-1]:8.4f}  grad norm {optim.result().metrics['grad_norm:mean']:.4f}")

state = trainer.save_state("final").result().path
trainer.load_state_with_optimizer(state).result()
print(f"state   {state}\nsampler {trainer.save_weights_for_sampler('final').result().path}")
assert losses[-1] < losses[0]*0.7, f"loss barely moved: {losses}"
