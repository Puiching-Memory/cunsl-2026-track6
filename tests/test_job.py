"""Linux-only checks that busy GPU jobs cannot launch or displace other work."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.run_job import acquire_gpu_lock

@unittest.skipUnless(sys.platform == "linux","Linux flock only")
class SharedLockTests(unittest.TestCase):
    def test_existing_lock_is_respected(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.lock"
            with path.open("a+") as other:
                fcntl.flock(other,fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch("tools.run_job.subprocess.run") as check:
                    with self.assertRaisesRegex(RuntimeError,"locked by another task"):
                        acquire_gpu_lock(path,0)
                    check.assert_not_called()
                # The incumbent still owns its lock after the rejected request.
                with path.open("a+") as probe:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(probe,fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_unlocked_but_occupied_gpu_is_rejected(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.lock"
            occupied = subprocess.CompletedProcess([],0,"123, other_track, 100 MiB\n","")
            with patch("tools.run_job.subprocess.run",return_value=occupied):
                with self.assertRaisesRegex(RuntimeError,"GPU is occupied"):
                    acquire_gpu_lock(path,0)
            with path.open("a+") as probe:
                fcntl.flock(probe,fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_lock_is_held_until_handle_closed(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.lock"
            idle = subprocess.CompletedProcess([],0,"","")
            with patch("tools.run_job.subprocess.run",return_value=idle):
                handle = acquire_gpu_lock(path,0)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["name"],"track6")
            with path.open("a+") as probe:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(probe,fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle.close()
                fcntl.flock(probe,fcntl.LOCK_EX | fcntl.LOCK_NB)

if __name__ == "__main__":
    unittest.main()
