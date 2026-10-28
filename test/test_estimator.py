import unittest
from unittest.mock import patch

from transformers import AutoConfig

from tinyrl.estimator import estimate


class FakeMeta:
  def __init__(self, parameter_count):
    self.parameter_count = parameter_count


class TestEstimator(unittest.TestCase):
  @patch("tinyrl.estimator.AutoConfig.from_pretrained")
  @patch("tinyrl.estimator.get_safetensors_metadata")
  def test_estimate(self, mock_safetensors, mock_config):
    mock_safetensors.return_value = FakeMeta({"BF16": 10000})  # 20000 base bytes

    config = AutoConfig.for_model(
      "qwen2",
      hidden_size=64,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=2,
      intermediate_size=128,
      max_position_embeddings=128,
      vocab_size=128,
      tie_word_embeddings=False,
    )
    mock_config.return_value = config

    budget = 80 * 1024**3
    overhead = int(0.75 * 1024**3)
    base = 20000

    # Full fine-tune: 10000 params * 16 bytes = 160000
    # No base_bytes added
    total, width = estimate("fake", "full", budget=budget)
    self.assertEqual(total, 160000 + overhead)
    self.assertEqual(width, 1)

    # vLLM: kv_heads=2, dim=16, seq=max_model_len=128, layers=2
    # bytes = 2(layers)*2(kv)*128(seq)*2(heads)*16(dim)*2(bytes) = 32768
    total, width = estimate("fake", "vllm", budget=budget, max_model_len=128)
    self.assertEqual(total, base + 32768 + overhead)

    # LoRA: run with rank=8
    # Attention + MLP = 16384
    # lm_head = 8 * (64 + 128) = 1536
    # Total = 17920 params * 16 bytes = 286720
    total, width = estimate("fake", "lora", runs=[{"rank": 8}], budget=budget)
    self.assertEqual(total, base + 286720 + overhead)

    # Multiple LoRA runs: 2 runs -> 2 * 286720 = 573440
    total, width = estimate("fake", "lora", runs=[{"rank": 8}, {"rank": 8}], budget=budget)
    self.assertEqual(total, base + 573440 + overhead)

    # Width check: budget forces width 2.
    # (total - overhead) / 2 + overhead <= budget
    target_budget = (base + 286720) // 2 + overhead + 10
    total, width = estimate("fake", "lora", runs=[{"rank": 8}], budget=target_budget)
    self.assertEqual(width, 2)

    # Width check: budget too small
    total, width = estimate("fake", "lora", runs=[{"rank": 8}], budget=overhead + 1)
    self.assertIsNone(width)


if __name__ == "__main__":
  unittest.main()
