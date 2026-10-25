import hashlib
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from tinyrl.payloads import INLINE_BYTES, PayloadStore


class TestPayloadStore(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.store = PayloadStore(self.directory.name)

  def test_durable_content_addressed_roundtrip_and_deduplication(self):
    body = bytes(range(256)) * (INLINE_BYTES // 256 + 1)
    with patch("tinyrl.payloads.os.fsync", wraps=os.fsync) as fsync:
      ref = self.store.put(body)
    self.assertEqual(ref, hashlib.sha256(body).hexdigest())
    self.assertGreaterEqual(fsync.call_count, 2)  # bytes and published directory entry
    inode = self.store.path(ref).stat().st_ino
    self.assertEqual(self.store.put(body), ref)
    self.assertEqual(self.store.path(ref).stat().st_ino, inode)
    self.assertEqual(self.store.get(ref, len(body)), body)
    other = self.store.put(body + b"different")
    self.assertNotEqual(other, ref)
    self.assertEqual(self.store.get(ref, len(body)), body)
    self.assertEqual(len(list(self.store.root.iterdir())), 2)

  def test_corruption_is_rejected_on_read_and_duplicate_publication(self):
    body = b"immutable training input"
    ref = self.store.put(body)
    self.store.path(ref).write_bytes(b"X" * len(body))
    with self.assertRaises(ValueError):
      self.store.get(ref, len(body))
    with self.assertRaises(ValueError):
      self.store.put(body)
    self.assertEqual(self.store.path(ref).read_bytes(), b"X" * len(body))
    self.assertEqual(len(list(self.store.root.iterdir())), 1)  # no abandoned temporary file

  def test_reference_and_size_validation_precedes_body_loading(self):
    body = b"small"
    ref = self.store.put(body)
    invalid_refs = ("../secret", "/absolute/path", "a" * 63, "a" * 65, "A" * 64, "x" * 64)
    original_open = Path.open

    @contextmanager
    def without_read(path, mode):
      with original_open(path, mode) as source:
        guarded = Mock(wraps=source)
        guarded.read.side_effect = AssertionError("must reject before loading bytes")
        yield guarded

    with patch.object(Path, "open", without_read):
      for invalid in invalid_refs:
        with self.subTest(ref=invalid), self.assertRaises(ValueError):
          self.store.get(invalid, len(body))
      for size in (-1, True, "5", len(body) - 1, len(body) + 1, 1 << 60):
        with self.subTest(size=size), self.assertRaises(ValueError):
          self.store.get(ref, size)

  def test_file_growth_during_read_cannot_load_an_unbounded_payload(self):
    body = b"small"
    ref = self.store.put(body)
    original_open, loaded = Path.open, []

    @contextmanager
    def growing_file(path, mode):
      with original_open(path, mode) as source:
        guarded = Mock(wraps=source)

        def read(*args):
          with original_open(path, "ab") as writer:
            writer.write(b"unexpected growth" * 4096)
          result = source.read(*args)
          loaded.append(len(result))
          return result

        guarded.read.side_effect = read
        yield guarded

    with patch.object(Path, "open", growing_file), self.assertRaises(ValueError):
      self.store.get(ref, len(body))
    self.assertEqual(loaded, [len(body) + 1])

  def test_failed_durability_does_not_publish_or_leave_temporary_files(self):
    with patch("tinyrl.payloads.os.fsync", side_effect=OSError("storage unavailable")):
      with self.assertRaises(OSError):
        self.store.put(b"unacknowledged bytes")
    self.assertEqual(list(self.store.root.iterdir()), [])


if __name__ == "__main__":
  unittest.main()
