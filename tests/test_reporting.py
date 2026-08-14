import csv
import tempfile
import unittest
from pathlib import Path

from image_matcher.reporting import write_candidates_csv, write_results_csv


class ReportingTest(unittest.TestCase):
    def test_write_results_csv_match_and_not_found_and_skipped(self):
        results = [
            {
                "reference": "ref1.jpg",
                "matched": "src1.jpg",
                "saved_as": "ref1.jpg",
                "good_matches": 250,
                "inliers": 230,
                "score": 0.9876,
                "stage1_candidate_count": 8,
                "winning_detector": "SIFT",
                "decision_reason": "ACCEPT_STRONG_GEOMETRY",
                "reject_reason": "",
            },
            {
                "reference": "ref2.jpg",
                "matched": None,
                "near_miss_file": "src2.jpg",
                "near_miss_good": 150,
                "near_miss_inliers": 100,
                "near_miss_score": 0.5,
                "stage1_diag": "no candidates",
                "stage1_candidate_count": 0,
                "winning_detector": "",
                "reject_reason": "REJECT_SCORE",
            },
            {
                "reference": "ref3.jpg",
                "skipped": True,
                "saved_as": "ref3.jpg",
                "matched": None,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "results.csv"
            write_results_csv(results, out_path)
            with open(out_path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))

        header, match_row, not_found_row, skipped_row = rows
        self.assertEqual(header[0], "Reference")

        self.assertEqual(match_row[:6], ["ref1.jpg", "src1.jpg", "ref1.jpg", "250", "230", "0.9876"])

        self.assertEqual(not_found_row[0], "ref2.jpg")
        self.assertEqual(not_found_row[1], "NOT_FOUND")
        self.assertEqual(not_found_row[6], "src2.jpg")
        self.assertEqual(not_found_row[13], "REJECT_NO_INLIERS")  # stage1_diag jelenlétekor a fallback ok

        self.assertEqual(skipped_row[:2], ["ref3.jpg", "SKIPPED_ALREADY_IN_FOUND"])

    def test_write_candidates_csv(self):
        candidates_detail = [
            {
                "reference": "ref1.jpg",
                "stage1_rank": 1,
                "stage1_score": 0.8,
                "candidate": "src1.jpg",
                "detector": "SIFT",
                "good_matches": 250,
                "inliers": 230,
                "inlier_ratio": 0.92,
                "stage2_score": 0.99,
                "success": True,
                "is_winner": True,
                "decision_reason": "ACCEPT_STRONG_GEOMETRY",
                "reject_reason": "",
            },
            {
                "reference": "ref1.jpg",
                "stage1_rank": None,
                "stage1_score": None,
                "candidate": "src2.jpg",
                "detector": "ORB",
                "good_matches": 50,
                "inliers": 10,
                "inlier_ratio": 0.2,
                "stage2_score": 0.1,
                "success": False,
                "is_winner": False,
                "reject_reason": "REJECT_INLIER_RATIO",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "candidates.csv"
            write_candidates_csv(candidates_detail, out_path)
            with open(out_path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))

        header, winner_row, loser_row = rows
        self.assertEqual(header[0], "Reference")

        self.assertEqual(winner_row[1], "1")
        self.assertEqual(winner_row[9], "IGEN")  # Success
        self.assertEqual(winner_row[10], "IGEN")  # IsWinner

        self.assertEqual(loser_row[1], "")  # stage1_rank hiányzik
        self.assertEqual(loser_row[9], "NEM")  # Success
        self.assertEqual(loser_row[10], "")  # IsWinner


if __name__ == "__main__":
    unittest.main()
