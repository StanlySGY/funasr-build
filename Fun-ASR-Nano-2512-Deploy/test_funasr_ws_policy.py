import unittest

from funasr_ws_policy import should_flush_final_online, should_run_online_chunk


class FunasrWsPolicyTest(unittest.TestCase):
    def test_skips_final_online_flush_for_fast_file_uploads(self):
        self.assertFalse(
            should_flush_final_online(
                mode="online",
                is_speaking=False,
                has_buffered_online_frames=True,
                skip_final_online_flush=True,
            )
        )

    def test_keeps_final_online_flush_for_realtime_online_streams(self):
        self.assertTrue(
            should_flush_final_online(
                mode="online",
                is_speaking=False,
                has_buffered_online_frames=True,
                skip_final_online_flush=False,
            )
        )

    def test_does_not_flush_without_buffered_online_frames(self):
        self.assertFalse(
            should_flush_final_online(
                mode="online",
                is_speaking=False,
                has_buffered_online_frames=False,
                skip_final_online_flush=False,
            )
        )

    def test_does_not_flush_non_online_modes(self):
        self.assertFalse(
            should_flush_final_online(
                mode="offline",
                is_speaking=False,
                has_buffered_online_frames=True,
                skip_final_online_flush=False,
            )
        )

    def test_defers_online_chunk_work_until_end_for_fast_file_uploads(self):
        self.assertFalse(
            should_run_online_chunk(
                mode="online",
                defer_online_until_end=True,
                reached_chunk_interval=True,
                is_final=False,
            )
        )

    def test_runs_online_chunk_work_for_realtime_online_streams(self):
        self.assertTrue(
            should_run_online_chunk(
                mode="online",
                defer_online_until_end=False,
                reached_chunk_interval=True,
                is_final=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
