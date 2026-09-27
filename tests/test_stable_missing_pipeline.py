"""
Integration tests for the stable-missing-set fix in smooth_compliance().

These tests exercise the FULL pipeline:
    smooth_compliance()  ->  log_violation()

Unlike the tests in test_violation_logging.py (which call log_violation()
directly with a pre-built missing_items list), these tests simulate the
real per-frame call pattern executed by run_detection().

Root-cause: smooth_compliance() computed smoothed_missing from
`required_ppe - current_detected_ppe` (a single frame), causing the
frozenset compared inside log_violation() to flicker every frame and
produce spurious PPE STATE CHANGE events.

Fix: per-item majority vote across the ppe_history window.
These tests verify the corrected behaviour end-to-end.
"""

import os
import tempfile
import unittest
import main

POLICY = {'helmet', 'safety-vest', 'shoes'}
TRACK_ID = 99


def _feed(track_id, frames_ppe, policy, snapshots_dir, log_file, event_counter):
    """Push a list of detected-PPE sets through smooth_compliance + log_violation.
    Returns (total_events_logged, event_counter_after).
    """
    events_logged = 0
    for detected in frames_ppe:
        status, _, missing = main.smooth_compliance(track_id, detected, required_ppe=policy)
        did_log, _, event_counter = main.log_violation(
            track_id=track_id,
            status=status,
            missing_items=missing,
            track_box=[10, 10, 100, 200],
            frame=None,
            save_dir=snapshots_dir,
            log_file=log_file,
            active_required_ppe=policy,
            detected_ppe=detected,
            event_counter=event_counter,
        )
        if did_log:
            events_logged += 1
    return events_logged, event_counter


class TestStableMissingPipeline(unittest.TestCase):

    def setUp(self):
        main.worker_state.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.snapshots_dir = os.path.join(self.tmp.name, 'snapshots')
        os.makedirs(self.snapshots_dir)
        self.log_file = os.path.join(self.tmp.name, 'violation_log.csv')
        self.counter = 0

    def tearDown(self):
        self.tmp.cleanup()

    def test_1_detection_flicker_produces_one_event(self):
        """Shoes flicker in/out; helmet consistently absent.
        Stable missing set = {helmet}. Exactly ONE event must be logged.
        """
        frames = [
            {'safety-vest'},
            {'safety-vest', 'shoes'},
            {'safety-vest'},
            {'safety-vest', 'shoes'},
            {'safety-vest', 'shoes'},
            {'safety-vest'},
            {'safety-vest', 'shoes'},
            {'safety-vest'},
        ]
        logged, self.counter = _feed(TRACK_ID, frames, POLICY, self.snapshots_dir, self.log_file, self.counter)
        self.assertEqual(logged, 1, "Shoe flicker must not produce duplicate events; expected exactly 1 event")

    def test_2_genuine_state_change_logs_new_event(self):
        """Stable {helmet} -> stable {helmet, shoes}: must produce exactly 2 events."""
        phase_a = [{'safety-vest', 'shoes'}] * 6
        phase_b = [{'safety-vest'}] * 6

        la, self.counter = _feed(TRACK_ID, phase_a, POLICY, self.snapshots_dir, self.log_file, self.counter)
        lb, self.counter = _feed(TRACK_ID, phase_b, POLICY, self.snapshots_dir, self.log_file, self.counter)

        self.assertEqual(la, 1, "Phase A must produce exactly 1 event")
        self.assertEqual(lb, 1, "Phase B must produce exactly 1 event for genuine state change")

    def test_3_same_stable_state_many_frames_one_event(self):
        """30 frames of consistent {helmet} missing -> exactly 1 event, never more."""
        frames = [{'safety-vest', 'shoes'}] * 30
        logged, self.counter = _feed(TRACK_ID, frames, POLICY, self.snapshots_dir, self.log_file, self.counter)
        self.assertEqual(logged, 1, "30 identical stable frames must produce exactly 1 event")

    def test_4_violation_then_safe_no_extra_row(self):
        """Missing {helmet} then fully SAFE: 1 event for violation, 0 for SAFE."""
        violation_frames = [{'safety-vest', 'shoes'}] * 6
        safe_frames      = [{'helmet', 'safety-vest', 'shoes'}] * 6

        lv, self.counter = _feed(TRACK_ID, violation_frames, POLICY, self.snapshots_dir, self.log_file, self.counter)
        ls, self.counter = _feed(TRACK_ID, safe_frames,      POLICY, self.snapshots_dir, self.log_file, self.counter)

        self.assertEqual(lv, 1, "Violation phase: exactly 1 event")
        self.assertEqual(ls, 0, "SAFE phase: 0 events (no violation row)")
        self.assertEqual(main.worker_state[TRACK_ID]['last_confirmed_status'], 'SAFE')

    def test_5_safe_then_re_violation_is_new_event(self):
        """{helmet} -> SAFE -> {helmet}: re-violation is a new event. Total = 2."""
        l1, self.counter = _feed(TRACK_ID, [{'safety-vest', 'shoes'}] * 6, POLICY, self.snapshots_dir, self.log_file, self.counter)
        l2, self.counter = _feed(TRACK_ID, [{'helmet', 'safety-vest', 'shoes'}] * 6, POLICY, self.snapshots_dir, self.log_file, self.counter)
        l3, self.counter = _feed(TRACK_ID, [{'safety-vest', 'shoes'}] * 6, POLICY, self.snapshots_dir, self.log_file, self.counter)

        self.assertEqual(l1, 1)
        self.assertEqual(l2, 0)
        self.assertEqual(l3, 1, "Re-violation after SAFE must be a new event")
        self.assertEqual(self.counter, 2, "Total event counter must be 2")

    def test_6_multiple_state_changes_three_events(self):
        """{helmet} -> {helmet,shoes} -> {shoes} -> SAFE: exactly 3 events, 0 for SAFE."""
        la, self.counter = _feed(TRACK_ID, [{'safety-vest', 'shoes'}] * 6, POLICY, self.snapshots_dir, self.log_file, self.counter)
        lb, self.counter = _feed(TRACK_ID, [{'safety-vest'}] * 6,          POLICY, self.snapshots_dir, self.log_file, self.counter)
        lc, self.counter = _feed(TRACK_ID, [{'safety-vest', 'helmet'}] * 6, POLICY, self.snapshots_dir, self.log_file, self.counter)
        ld, self.counter = _feed(TRACK_ID, [{'helmet', 'safety-vest', 'shoes'}] * 6, POLICY, self.snapshots_dir, self.log_file, self.counter)

        self.assertEqual(la, 1, "State A: 1 event")
        self.assertEqual(lb, 1, "State B: 1 event for state change")
        self.assertEqual(lc, 1, "State C: 1 event for state change")
        self.assertEqual(ld, 0, "SAFE: 0 events")
        self.assertEqual(self.counter, 3, "Total: exactly 3 events")

    def test_7_csv_has_no_duplicate_stable_rows(self):
        """10 frames of stable {helmet} missing: CSV must have exactly 1 data row."""
        _feed(TRACK_ID, [{'safety-vest', 'shoes'}] * 10, POLICY, self.snapshots_dir, self.log_file, self.counter)

        with open(self.log_file) as f:
            data_lines = [l for l in f.readlines() if l.strip() and not l.startswith('Date')]

        self.assertEqual(len(data_lines), 1,
                         f"CSV must have exactly 1 data row, found {len(data_lines)}: {data_lines}")


if __name__ == '__main__':
    unittest.main()
