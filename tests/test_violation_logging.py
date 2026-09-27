import unittest
import os
import tempfile
import main

class TestViolationLogging(unittest.TestCase):
    def setUp(self):
        main.worker_state.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.snapshots_dir = os.path.join(self.temp_dir.name, "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)
        self.log_file = os.path.join(self.temp_dir.name, "violation_log.csv")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initial_violation_logged(self):
        policy = {'helmet', 'safety-vest', 'shoes'}
        detected = {'safety-vest'}  # missing helmet, shoes
        logged, rec, cnt = main.log_violation(
            track_id=1,
            status='PARTIAL PPE',
            missing_items=['helmet', 'shoes'],
            track_box=[10, 10, 100, 200],
            frame=None,
            save_dir=self.snapshots_dir,
            log_file=self.log_file,
            active_required_ppe=policy,
            detected_ppe=detected,
            event_counter=0,
            current_time=100.0
        )
        self.assertTrue(logged, "Initial violation must be logged")
        self.assertEqual(rec['event'], 'INITIAL VIOLATION')
        self.assertEqual(rec['person'], 'Person 1')
        self.assertEqual(cnt, 1)

    def test_unchanged_state_never_repeated(self):
        policy = {'helmet', 'safety-vest', 'shoes'}
        detected = {'safety-vest'}
        # Initial event
        logged1, _, cnt1 = main.log_violation(
            track_id=2, status='PARTIAL PPE', missing_items=['helmet', 'shoes'],
            track_box=[10, 10, 100, 200], frame=None, save_dir=self.snapshots_dir, log_file=self.log_file,
            active_required_ppe=policy, detected_ppe=detected, event_counter=0, current_time=100.0
        )
        self.assertTrue(logged1)
        self.assertEqual(cnt1, 1)

        # Same state 15 seconds later -> NO LOG
        logged2, _, cnt2 = main.log_violation(
            track_id=2, status='PARTIAL PPE', missing_items=['helmet', 'shoes'],
            track_box=[10, 10, 100, 200], frame=None, save_dir=self.snapshots_dir, log_file=self.log_file,
            active_required_ppe=policy, detected_ppe=detected, event_counter=cnt1, current_time=115.0
        )
        self.assertFalse(logged2, "Identical violation state must NOT be logged again")
        self.assertEqual(cnt2, 1)

        # Same state 500 seconds later -> NO LOG (no 30-second cooldown repeated logs!)
        logged3, _, cnt3 = main.log_violation(
            track_id=2, status='PARTIAL PPE', missing_items=['helmet', 'shoes'],
            track_box=[10, 10, 100, 200], frame=None, save_dir=self.snapshots_dir, log_file=self.log_file,
            active_required_ppe=policy, detected_ppe=detected, event_counter=cnt2, current_time=600.0
        )
        self.assertFalse(logged3, "Identical violation state must NOT be logged even after extended time")
        self.assertEqual(cnt3, 1)

    def test_state_change_logged(self):
        policy = {'helmet', 'safety-vest', 'shoes'}
        # 1. Initial: missing shoes
        logged1, rec1, cnt1 = main.log_violation(
            track_id=3, status='PARTIAL PPE', missing_items=['shoes'],
            track_box=[10, 10, 100, 200], frame=None, save_dir=self.snapshots_dir, log_file=self.log_file,
            active_required_ppe=policy, detected_ppe={'helmet', 'safety-vest'}, event_counter=0, current_time=100.0
        )
        self.assertTrue(logged1)
        self.assertEqual(rec1['event'], 'INITIAL VIOLATION')

        # 2. Worker now removes helmet (missing helmet + shoes) -> STATE CHANGE!
        logged2, rec2, cnt2 = main.log_violation(
            track_id=3, status='PARTIAL PPE', missing_items=['helmet', 'shoes'],
            track_box=[10, 10, 100, 200], frame=None, save_dir=self.snapshots_dir, log_file=self.log_file,
            active_required_ppe=policy, detected_ppe={'safety-vest'}, event_counter=cnt1, current_time=110.0
        )
        self.assertTrue(logged2, "Different violation state must be logged")
        self.assertEqual(rec2['event'], 'PPE STATE CHANGE')
        self.assertEqual(cnt2, 2)

    def test_transition_to_safe_and_re_violation(self):
        policy = {'helmet', 'shoes'}
        # 1. Initial violation: missing shoes
        logged1, _, cnt1 = main.log_violation(
            track_id=4, status='PARTIAL PPE', missing_items=['shoes'],
            track_box=[10, 10, 100, 200], frame=None, save_dir=self.snapshots_dir, log_file=self.log_file,
            active_required_ppe=policy, detected_ppe={'helmet'}, event_counter=0, current_time=100.0
        )
        self.assertTrue(logged1)

        # 2. Worker puts on shoes -> becomes SAFE
        logged2, _, cnt2 = main.log_violation(
            track_id=4, status='SAFE', missing_items=[],
            track_box=[10, 10, 100, 200], frame=None, save_dir=self.snapshots_dir, log_file=self.log_file,
            active_required_ppe=policy, detected_ppe={'helmet', 'shoes'}, event_counter=cnt1, current_time=120.0
        )
        self.assertFalse(logged2, "SAFE state must never log violation row")
        self.assertEqual(cnt2, 1)

        # 3. Later, worker removes shoes again -> MUST LOG NEW EVENT
        logged3, rec3, cnt3 = main.log_violation(
            track_id=4, status='PARTIAL PPE', missing_items=['shoes'],
            track_box=[10, 10, 100, 200], frame=None, save_dir=self.snapshots_dir, log_file=self.log_file,
            active_required_ppe=policy, detected_ppe={'helmet'}, event_counter=cnt2, current_time=150.0
        )
        self.assertTrue(logged3, "Violation after being SAFE must be logged")
        self.assertEqual(rec3['event'], 'PPE STATE CHANGE')
        self.assertEqual(cnt3, 2)

    def test_safe_worker_never_logged(self):
        policy = {'helmet', 'shoes'}
        logged, _, cnt = main.log_violation(
            track_id=5, status='SAFE', missing_items=[],
            track_box=[10, 10, 100, 200], frame=None, save_dir=self.snapshots_dir, log_file=self.log_file,
            active_required_ppe=policy, detected_ppe={'helmet', 'shoes'}, event_counter=0, current_time=100.0
        )
        self.assertFalse(logged, "Worker entering SAFE must not log violation")
        self.assertEqual(cnt, 0)

if __name__ == "__main__":
    unittest.main()
