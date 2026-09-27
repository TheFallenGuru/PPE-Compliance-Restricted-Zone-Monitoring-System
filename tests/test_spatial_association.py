import unittest

def compute_overlap_ratio(person_box, ppe_box):
    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = ppe_box
    ix1 = max(px1, ox1); iy1 = max(py1, oy1); ix2 = min(px2, ox2); iy2 = min(py2, oy2)
    inter_w = max(0, ix2 - ix1); inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h
    ppe_area = max(1, (ox2 - ox1) * (oy2 - oy1))
    return inter_area / ppe_area

def assign_ppe_to_persons(persons, ppe_detections, overlap_threshold=0.3):
    person_ppe = {i: set() for i in range(len(persons))}
    for ppe_box, class_name, conf in ppe_detections:
        best_ratio = overlap_threshold
        best_person = None
        for i, person_box in enumerate(persons):
            ratio = compute_overlap_ratio(person_box, ppe_box)
            if ratio > best_ratio:
                best_ratio = ratio
                best_person = i
        if best_person is not None:
            person_ppe[best_person].add(class_name)
    return person_ppe

class TestSpatialAssociation(unittest.TestCase):
    def test_complete_containment(self):
        person = [100, 100, 300, 400]
        helmet = [150, 110, 250, 180]
        ratio = compute_overlap_ratio(person, helmet)
        self.assertAlmostEqual(ratio, 1.0, places=2)

    def test_no_overlap(self):
        person = [100, 100, 300, 400]
        ppe = [500, 500, 600, 600]
        ratio = compute_overlap_ratio(person, ppe)
        self.assertEqual(ratio, 0.0)

    def test_winner_takes_all_no_sharing(self):
        # Two workers side-by-side with overlapping bounding boxes
        p1 = [100, 100, 250, 400]
        p2 = [200, 100, 350, 400]
        # Helmet is strictly in p1 area [120..180]
        helmet_p1 = ([120, 110, 180, 170], 'helmet', 0.9)
        # Vest is strictly in p2 area [280..340]
        vest_p2 = ([280, 180, 340, 280], 'safety-vest', 0.85)

        mapping = assign_ppe_to_persons([p1, p2], [helmet_p1, vest_p2])
        self.assertIn('helmet', mapping[0])
        self.assertNotIn('helmet', mapping[1])
        self.assertIn('safety-vest', mapping[1])
        self.assertNotIn('safety-vest', mapping[0])

if __name__ == "__main__":
    unittest.main()
