"""Counterexamples distinguishing achievable shortlist order from missing positives."""
import unittest

import numpy as np
from track6.diagnostics import shortlist_bounds


class ShortlistBoundTests(unittest.TestCase):
    def test_multiple_positives_outside_shortlist_still_count(self):
        rel=np.zeros((1,15),dtype=bool)
        rel[0,[2,8,11]]=True
        report=shortlist_bounds(rel,[10,15])
        self.assertEqual(report['actual']['R@1'],0)
        self.assertAlmostEqual(report['actual']['mAP@10'],(1/3+2/9)/3)
        self.assertEqual(report['shortlists']['10']['oracle_R@1'],1)
        self.assertAlmostEqual(report['shortlists']['10']['oracle_mAP@10'],2/3)
        self.assertEqual(report['shortlists']['15']['oracle_mAP@10'],1)
        rel[:,:10]=False
        missing=shortlist_bounds(rel,[10,15])['shortlists']['10']
        self.assertEqual(missing['oracle_direction_score'],0)
        self.assertEqual(missing['queries_with_no_positive'],1)

    def test_two_ap_denominators_differ_above_ten_positives(self):
        rel=np.zeros((1,15),dtype=bool);rel[0,:12]=True
        truncated=shortlist_bounds(rel,[10],denominator='min_relevant_k')
        all_positive=shortlist_bounds(rel,[10],denominator='all_relevant')
        self.assertEqual(truncated['actual']['mAP@10'],1)
        self.assertEqual(truncated['shortlists']['10']['oracle_mAP@10'],1)
        self.assertAlmostEqual(all_positive['actual']['mAP@10'],10/12)
        self.assertAlmostEqual(all_positive['shortlists']['10']['oracle_mAP@10'],10/12)

    def test_undefined_bounds_and_bad_inputs_are_rejected(self):
        for rel,cutoffs in [(np.zeros((1,15),bool),[10]),(np.ones((1,9),bool),[10]),
                            (np.ones((1,15),int),[10]),(np.ones((1,15),bool),[9]),
                            (np.ones((1,15),bool),[10,10]),(np.ones((1,15),bool),[16]),
                            (np.ones((1,15),bool),[10.5])]:
            with self.subTest(cutoffs=cutoffs),self.assertRaises(ValueError):
                shortlist_bounds(rel,cutoffs)


if __name__=='__main__':
    unittest.main()
