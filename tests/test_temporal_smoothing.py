import unittest
import main

class TestTemporalSmoothing(unittest.TestCase):
    def setUp(self):
        main.worker_state.clear()

    def test_one_frame_glitch_suppressed(self):
        all_ppe = {'helmet', 'safety-vest', 'gloves', 'shoes', 'face-mask'}
        for _ in range(4):
            main.smooth_compliance('w1', all_ppe)
        # 1-frame detection dropout
        st, _, _ = main.smooth_compliance('w1', set())
        self.assertEqual(st, 'SAFE', "1-frame detection dropout must be smoothed to SAFE")

    def test_tie_breaking_severity(self):
        # Exactly 1 SAFE and 1 UNSAFE frame -> severity tie-break should pick UNSAFE
        all_ppe = {'helmet', 'safety-vest', 'gloves', 'shoes', 'face-mask'}
        main.smooth_compliance('w2', all_ppe)
        st, _, _ = main.smooth_compliance('w2', set())
        self.assertEqual(st, 'UNSAFE', "Tie-break must pick higher severity (UNSAFE)")

    def test_custom_checklist_temporal_smoothing(self):
        custom_req = {'gloves', 'shoes', 'glasses'}
        # 4 frames with custom required PPE detected
        for _ in range(4):
            main.smooth_compliance('w3', {'gloves', 'shoes', 'glasses'}, required_ppe=custom_req)
        # 1 frame dropout
        st, _, missing = main.smooth_compliance('w3', {'gloves'}, required_ppe=custom_req)
        self.assertEqual(st, 'SAFE', "Glitch should be smoothed to SAFE against custom checklist")
        self.assertEqual(missing, [])

        # Now feed 4 UNSAFE frames (missing glasses & shoes)
        for _ in range(4):
            st, _, missing = main.smooth_compliance('w3', {'gloves'}, required_ppe=custom_req)
        self.assertEqual(st, 'PARTIAL PPE')
        self.assertEqual(missing, ['glasses', 'shoes'])

if __name__ == "__main__":
    unittest.main()
