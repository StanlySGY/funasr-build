import unittest

from funasr_ws_policy import should_flush_final_online


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


if __name__ == "__main__":
    unittest.main()
