import torch


def cross_entropy_loss(target_logprobs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
  return -target_logprobs * weights


def importance_sampling_loss(
  target_logprobs: torch.Tensor,
  weights: torch.Tensor,
  old_logprobs: torch.Tensor,
  advantages: torch.Tensor,
) -> torch.Tensor:
  ratio = policy_ratio(target_logprobs, old_logprobs)
  elementwise_loss = -(ratio * advantages) * weights
  return torch.nan_to_num(elementwise_loss, nan=0.0, posinf=0.0, neginf=0.0)


def ppo_loss(
  target_logprobs: torch.Tensor,
  weights: torch.Tensor,
  old_logprobs: torch.Tensor,
  advantages: torch.Tensor,
  loss_config: dict | None,
) -> torch.Tensor:
  diff = torch.clamp(target_logprobs - old_logprobs, min=-20.0, max=20.0)
  ratio = torch.exp(diff)
  epsilon = loss_config.get("clip_range", 0.2) if loss_config else 0.2
  surr1 = ratio * advantages
  surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
  elementwise_objective = torch.min(surr1, surr2)
  kl_coeff = loss_config.get("kl_coeff", 0.0) if loss_config else 0.0
  if kl_coeff > 0:
    kl = (ratio - 1) - diff
    elementwise_objective = elementwise_objective - kl_coeff * kl
  return -(elementwise_objective * weights)


def policy_ratio(target_logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
  return torch.exp(torch.clamp(target_logprobs - ref_logprobs, min=-20.0, max=20.0))
