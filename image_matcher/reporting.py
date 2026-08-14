"""CSV kimenetek: referenciánkénti összegzés és jelölt×detektor-szintű részletes napló."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def write_results_csv(results: List[Dict], out_path: Path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Reference", "MatchedFile", "SavedAs", "GoodMatches", "Inliers", "Score",
                          "NearMissFile", "NearMissGood", "NearMissInliers", "NearMissScore",
                          "Stage1Diag", "Stage1Candidates", "WinningDetector",
                          "DecisionReason", "RejectReason"])
        for r in results:
            if r.get("skipped"):
                writer.writerow([r["reference"], "SKIPPED_ALREADY_IN_FOUND", r.get("saved_as", "")])
            elif r["matched"] is None:
                writer.writerow([
                    r["reference"], "NOT_FOUND", "", "", "", "",
                    r.get("near_miss_file", ""),
                    r.get("near_miss_good", ""),
                    r.get("near_miss_inliers", ""),
                    r.get("near_miss_score", ""),
                    r.get("stage1_diag", ""),
                    r.get("stage1_candidate_count", ""),
                    r.get("winning_detector", ""),
                    r.get("decision_reason", "REJECT_NO_INLIERS" if r.get("stage1_diag") else ""),
                    r.get("reject_reason", "")
                ])
            else:
                writer.writerow([
                    r["reference"],
                    r["matched"],
                    r.get("saved_as", ""),
                    r["good_matches"],
                    r["inliers"],
                    f"{r['score']:.4f}",
                    "", "", "", "", "",
                    r.get("stage1_candidate_count", ""),
                    r.get("winning_detector", ""),
                    r.get("decision_reason", ""),
                    r.get("reject_reason", "")
                ])


def write_candidates_csv(candidates_detail: List[Dict], out_path: Path):
    """
    Jelölt×detektor-szintű RÉSZLETES napló – minden a 2. körben ténylegesen
    kipróbált (jelölt, detektor) kombináció saját sorban, a Stage-1 rangot/
    score-t, a Stage-2 minden mérőszámát, és a pontos bukási okot is
    tartalmazva.
    """
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Reference", "Stage1Rank", "Stage1Score", "Candidate", "Detector",
            "GoodMatches", "Inliers", "InlierRatio", "Stage2Score",
            "Success", "IsWinner", "DecisionReason", "RejectReason"
        ])
        for c in candidates_detail:
            writer.writerow([
                c["reference"],
                c["stage1_rank"] if c["stage1_rank"] is not None else "",
                f"{c['stage1_score']:.4f}" if c["stage1_score"] is not None else "",
                c["candidate"],
                c["detector"],
                c["good_matches"],
                c["inliers"],
                f"{c['inlier_ratio']:.4f}",
                f"{c['stage2_score']:.4f}",
                "IGEN" if c["success"] else "NEM",
                "IGEN" if c["is_winner"] else "",
                c.get("decision_reason", ""),
                c["reject_reason"]
            ])
