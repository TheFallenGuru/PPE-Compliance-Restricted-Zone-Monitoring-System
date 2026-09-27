import unittest
import main

class TestComplianceClassifier(unittest.TestCase):
    def test_default_policy_requirements(self):
        # Default policy must contain the 5 standard PPE items
        expected_default = {'helmet', 'safety-vest', 'gloves', 'shoes', 'face-mask'}
        self.assertEqual(main.DEFAULT_REQUIRED_PPE, expected_default)

    def test_all_required_ppe_safe(self):
        detected = {'helmet', 'safety-vest', 'gloves', 'shoes', 'face-mask'}
        status, color, missing = main.classify_compliance(detected)
        self.assertEqual(status, 'SAFE')
        self.assertEqual(missing, [])

    def test_partial_ppe(self):
        detected = {'helmet', 'safety-vest'}
        status, color, missing = main.classify_compliance(detected)
        self.assertEqual(status, 'PARTIAL PPE')
        self.assertEqual(missing, ['face-mask', 'gloves', 'shoes'])

    def test_no_ppe_unsafe(self):
        detected = set()
        status, color, missing = main.classify_compliance(detected)
        self.assertEqual(status, 'UNSAFE')
        self.assertEqual(len(missing), 5)

    def test_custom_policy_all_detected_safe(self):
        required = {'gloves', 'shoes', 'glasses'}
        detected = {'gloves', 'shoes', 'glasses'}
        status, color, missing = main.classify_compliance(detected, required_ppe=required)
        self.assertEqual(status, 'SAFE')
        self.assertEqual(missing, [])

    def test_custom_policy_missing_one_partial(self):
        required = {'gloves', 'shoes', 'glasses'}
        detected = {'gloves', 'shoes'}
        status, color, missing = main.classify_compliance(detected, required_ppe=required)
        self.assertEqual(status, 'PARTIAL PPE')
        self.assertEqual(missing, ['glasses'])

    def test_custom_policy_missing_all_unsafe(self):
        required = {'gloves', 'shoes', 'glasses'}
        detected = set()
        status, color, missing = main.classify_compliance(detected, required_ppe=required)
        self.assertEqual(status, 'UNSAFE')
        self.assertEqual(missing, ['glasses', 'gloves', 'shoes'])

    def test_extra_non_required_items_ignored_for_compliance(self):
        # Extra detectable PPE (safety-vest, glasses) not in required set does not fail compliance
        required = {'helmet', 'gloves'}
        detected = {'helmet', 'gloves', 'safety-vest', 'glasses'}
        status, color, missing = main.classify_compliance(detected, required_ppe=required)
        self.assertEqual(status, 'SAFE')
        self.assertEqual(missing, [])

    def test_parse_required_ppe_helper(self):
        # String format parsing
        res1 = main.parse_required_ppe("helmet,gloves,glasses")
        self.assertEqual(res1, {'helmet', 'gloves', 'glasses'})

        # Semicolon/space parsing
        res2 = main.parse_required_ppe("shoes; safety-vest face-mask")
        self.assertEqual(res2, {'shoes', 'safety-vest', 'face-mask'})

        # Invalid/empty falls back to default
        res3 = main.parse_required_ppe(None)
        self.assertEqual(res3, main.DEFAULT_REQUIRED_PPE)

        res4 = main.parse_required_ppe("person,invalid-item")
        self.assertEqual(res4, main.DEFAULT_REQUIRED_PPE)

if __name__ == "__main__":
    unittest.main()
