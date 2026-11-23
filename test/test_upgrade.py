import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from test.helpers import BusTest
from tinyrl.control.ledger import Ledger
from tinyrl.control.upgrade import checkpoint_meta
from tinyrl.types import NodeStatus, Op, Reply, Role, Run

OK = Reply(True, "application/json", b"{}")
LEGACY_DB = Path(__file__).with_name("fixtures") / "bus_7afae95.sql"


class TestPopulatedUpgrade(BusTest):
  def test_future_schema_is_rejected_without_writes(self):
    future = f"{self._db_dir.name}/future.db"
    with closing(sqlite3.connect(future)) as conn:
      conn.executescript("CREATE TABLE future(payload); PRAGMA user_version=2")
      before = conn.execute("SELECT sql FROM sqlite_master").fetchall()
      with self.assertRaisesRegex(ValueError, "unsupported ledger database version 2"):
        Ledger(future)
      self.assertEqual(conn.execute("SELECT sql FROM sqlite_master").fetchall(), before)
      self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0], "delete")

  async def test_old_wal_reader_and_malformed_load(self):
    old = f"{self._db_dir.name}/legacy-wal.db"
    with closing(sqlite3.connect(old)) as reader:
      reader.execute("PRAGMA journal_mode=WAL")
      reader.executescript(LEGACY_DB.read_text())
      reader.execute("UPDATE kv SET v=NULL WHERE ns='blobs' AND k='c:t/2700000000000-7'")
      reader.commit()
      reader.execute("BEGIN")
      self.assertEqual(reader.execute("PRAGMA user_version").fetchone()[0], 0)
      migrated = Ledger(old)
      try:
        self.assertEqual(reader.execute("PRAGMA user_version").fetchone()[0], 0)
        self.assertEqual(migrated.db.conn.execute("PRAGMA user_version").fetchone()[0], 1)
        migrated.expire_node("node", "boot")
        plan = await migrated.take("replacement", "new-boot", NodeStatus(1))
        self.assertFalse((await migrated.wait_for_reply("t/2700000000000-7", 0)).ok)
        self.assertEqual(plan.grants[0].work[0].cmd.request_id, "t/2700000000000-8")
      finally:
        await migrated.aclose()

  @patch("time.time", return_value=2_700_000_000.0)  # fixture's clock, including its reply and dedupe deadlines
  async def test_upgrade_from_bus_7afae95(self, _clock):
    await self.bus.aclose()
    with closing(sqlite3.connect(self.db_path)) as conn:
      conn.executescript("DROP TABLE kv; PRAGMA user_version=0;\n" + LEGACY_DB.read_text())
    self.bus = Ledger(self.db_path)
    done, claimed, queued, stable = [self.bus.seen_request("t", seq) for seq in (1, 2, 3, 4)]
    self.assertEqual(await self.bus.wait_for_reply(done, 0), Reply(True, "text/plain", b"done"))
    self.assertEqual(self.bus.get_session("test-session")["metadata"], {"sdk": "0.39"})
    self.assertEqual(self.bus.get_run("s")["status"], "deleted")
    self.assertEqual(self.bus.get_run("t")["steps"], 1)
    self.assertEqual(self.bus.queued_and_in_flight("t")["in_flight"][0]["request_id"], claimed)
    row = self.bus.trains.cmds[claimed]
    self.assertEqual(row.owner[1:], ("node", "boot"))
    self.assertEqual(row.block, (0,))
    self.assertEqual(len(row.version), 32)
    self.assertEqual(self.bus.trains.cmd(stable).stability, {"sample/1:0": b"top-k"})

    # The old worker was executing the save. Expiry fails it once, preserving the untouched FIFO tail.
    self.bus.expire_node("node", "boot")
    self.assertFalse((await self.bus.wait_for_reply(claimed, 0)).ok)
    self.assertEqual(self.bus.get_run("t")["status"], "lost")
    load = self.bus.trains.heads({"t"})["t"]
    self.assertEqual((load.cmd.request_id, load.uri), (queued, "tinker://t/weights/model"))
    self.assertEqual(self.bus.checkpoint_version(load.uri), "f" * 32)
    # The Bus persisted the *actual* LoRA targets in the checkpoint object, including omitted/tied heads.
    from tinyrl.storage.store import Store

    with patch("tinyrl.storage.store.CKPT", self._db_dir.name):
      Store().put({}, {"base": "test-model", "rank": 8, "targets": ["q_proj"]}, version="f" * 32)
      meta = await checkpoint_meta(self.bus.get_checkpoint(load.uri))
      self.assertEqual((meta["base"], meta["rank"]), ("test-model", 8))
      self.assertEqual(meta["flags"], {"train_unembed": False, "train_attn": True, "train_mlp": False})
    await self.bus.aclose()
    self.bus = Ledger(self.db_path)
    self.assertEqual(self.bus.trains.heads({"t"})["t"].cmd.request_id, queued)
    self.assertEqual(self.bus.seen_request("t", 1), done)

  async def test_new_owner_fails_old_pending_command_and_keeps_the_queued_tail(self):
    self.bus.create_run(Run("t", "base", 8, role=Role.TRAINER))
    first = self.bus.submit_command("t", Op.OPTIM_STEP, b"\xff")
    second = self.bus.submit_command("t", Op.OPTIM_STEP, b"\xff")
    [a1] = (await self.bus.take("n1", "b1", NodeStatus(1))).grants
    [w1] = a1.work
    self.assertEqual(w1.cmd.request_id, first)

    self.bus.nodes["n1"].until = 0.0
    [a2] = (await self.bus.take("n2", "b2", NodeStatus(1))).grants
    self.assertFalse((await self.bus.wait_for_reply(first, 0)).ok)
    self.assertEqual(self.bus.get_run("t")["failed"], 1)
    [w2] = a2.work
    self.assertEqual((w2.cmd.request_id, w2.cmd.body), (second, b"\xff"))
    self.bus.finish("n2", "b2", a2.worker, w2.cmd.request_id, OK, mutated=True)
    self.assertEqual(await self.bus.wait_for_reply(second, 0), OK)

  async def test_sampler_recovers_abandoned_claim_alongside_fresh_entries(self):
    self.bus.create_run(Run("s", "base", 8, role=Role.SAMPLER))
    first = self.bus.submit_command("s", Op.SAMPLE, b"{}")
    [a1] = (await self.bus.take("n1", "b1", NodeStatus(1))).grants
    [w1] = a1.work
    self.assertEqual(w1.cmd.request_id, first)

    second = self.bus.submit_command("s", Op.SAMPLE, b"{}")
    self.bus.nodes["n1"].until = 0.0
    [a2] = (await self.bus.take("n2", "b2", NodeStatus(1))).grants
    claimed = a2.work
    self.assertEqual([w.cmd.request_id for w in claimed], [first, second])
    for w in claimed:
      self.bus.finish("n2", "b2", a2.worker, w.cmd.request_id, OK, attempt=w.attempt)
    self.assertEqual(await self.bus.wait_for_reply(first, 0), OK)
    self.assertEqual(await self.bus.wait_for_reply(second, 0), OK)
    self.assertEqual(self.bus.queued_and_in_flight("s"), {"live": True, "queued": 0, "in_flight": []})

  async def test_persisted_metadata_and_trainer_replies_survive_db_reopen(self):
    self.bus.heartbeat_session("sess-1", {"sdk": "0.39"})
    self.bus.create_run(Run("t", "base", 8, session="sess-1", role=Role.TRAINER))
    self.seed_checkpoint("tinker://t/weights/v1", 42, version="ver-1")
    req = self.bus.submit_command("t", Op.OPTIM_STEP, b"{}", seq_id=7)
    [a1] = (await self.bus.take("n1", "b1", NodeStatus(1))).grants
    [w] = a1.work
    self.bus.finish("n1", "b1", a1.worker, w.cmd.request_id, OK, mutated=True)
    await self.bus.aclose()

    self.bus = Ledger(self.db_path)
    self.assertEqual(self.bus.get_session("sess-1")["metadata"], {"sdk": "0.39"})
    self.assertEqual(self.bus.checkpoint_version("tinker://t/weights/v1"), "ver-1")
    self.assertEqual(await self.bus.wait_for_reply(req, 0), OK)
    self.assertEqual(self.bus.seen_request("t", 7), req)
