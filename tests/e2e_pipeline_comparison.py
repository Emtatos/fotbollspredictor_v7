"""
End-to-end comparison: new scanner pipeline vs legacy parser.
Runs both on 3 real coupon images and reports results.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner_pipeline import run_scanner_pipeline, ScannerResult
from coupon_image_parser import parse_coupon_image

# Ground truth for each coupon image
GROUND_TRUTH = {
    "coupon_clean_13matches.png": {
        "expected_rows": 13,
        "teams": [
            ("Arsenal", "Liverpool"),
            ("Man Utd", "Chelsea"),
            ("Wolves", "Brighton"),
            ("Newcastle", "Tottenham"),
            ("Aston Villa", "West Ham"),
            ("Everton", "Nott'm Forest"),
            ("Bournemouth", "Crystal Palace"),
            ("Fulham", "Brentford"),
            ("Leicester", "Ipswich"),
            ("Southampton", "Leeds"),
            ("Sheff Utd", "QPR"),
            ("Derby", "Preston"),
            ("Luton", "Plymouth"),
        ],
        "canonical_teams_expected": {
            "Man Utd": "Manchester United",
            "Wolves": "Wolverhampton Wanderers",
            "Brighton": "Brighton & Hove Albion",
            "Newcastle": "Newcastle United",
            "Nott'm Forest": "Nottingham Forest",
            "Leicester": "Leicester City",
            "Sheff Utd": "Sheffield United",
            "QPR": "Queens Park Rangers",
            "Derby": "Derby County",
            "Luton": "Luton Town",
            "Plymouth": "Plymouth Argyle",
            "Preston": "Preston North End",
            "Leeds": "Leeds United",
            "Ipswich": "Ipswich Town",
            "West Ham": "West Ham United",
            "Tottenham": "Tottenham Hotspur",
        },
        "has_all_streck": True,
        "has_all_odds": True,
    },
    "coupon_allsvenskan_partial.png": {
        "expected_rows": 8,
        "teams": [
            ("AIK", "Djurgarden"),
            ("Hammarby", "Malmo FF"),
            ("IFK Goteborg", "IFK Norrkoping"),
            ("IF Elfsborg", "Hacken"),
            ("Kalmar FF", "Mjallby"),
            ("Sirius", "Varnamo"),
            ("Brommapojkarna", "Halmstad"),
            ("Degerfors", "Sundsvall"),
        ],
        "has_all_streck": True,
        "has_all_odds": False,  # rows 5,6 have no odds
    },
    "coupon_noisy_rotated.png": {
        "expected_rows": 6,
        "teams": [
            ("Man City", "Nott'm Forest"),
            ("Sheff Wed", "West Brom"),
            ("Brighton", "Wolverhampton"),
            ("Spurs", "Bournemouth"),
            ("Blackburn", "Stockport"),
            ("Cambridge Utd", "Charlton"),
        ],
        "canonical_teams_expected": {
            "Man City": "Manchester City",
            "Nott'm Forest": "Nottingham Forest",
            "Sheff Wed": "Sheffield Wednesday",
            "West Brom": "West Bromwich Albion",
            "Brighton": "Brighton & Hove Albion",
            "Wolverhampton": "Wolverhampton Wanderers",
            "Spurs": "Tottenham Hotspur",
            "Blackburn": "Blackburn Rovers",
            "Stockport": "Stockport County",
            "Cambridge Utd": "Cambridge United",
            "Charlton": "Charlton Athletic",
        },
        "has_all_streck": True,
        "has_all_odds": True,
    },
}


def _normalize_for_compare(name: str) -> str:
    """Normalize team name for comparison."""
    return name.lower().strip().replace("'", "'").replace("\u2019", "'")


def _team_match(extracted: str, expected: str, canonical_map: dict = None) -> bool:
    """Check if extracted team matches expected (or its canonical form)."""
    e = _normalize_for_compare(extracted)
    exp = _normalize_for_compare(expected)

    if e == exp:
        return True

    # Check canonical mapping
    if canonical_map and expected in canonical_map:
        canonical = _normalize_for_compare(canonical_map[expected])
        if e == canonical:
            return True

    # Fuzzy substring check
    if exp in e or e in exp:
        return True

    return False


def evaluate_pipeline_result(scanner_result: ScannerResult, ground_truth: dict, label: str) -> dict:
    """Evaluate scanner pipeline result against ground truth."""
    gt = ground_truth
    result = {
        "label": label,
        "expected_rows": gt["expected_rows"],
        "extracted_rows": scanner_result.total_rows,
        "ok_rows": scanner_result.ok_rows,
        "uncertain_rows": scanner_result.uncertain_rows,
        "failed_rows": scanner_result.failed_rows,
        "teams_correct": 0,
        "teams_total": gt["expected_rows"] * 2,
        "teams_mapped": 0,
        "streck_present": 0,
        "odds_present": 0,
        "errors": [],
        "issues_summary": [],
    }

    canonical_map = gt.get("canonical_teams_expected", {})

    for i, row in enumerate(scanner_result.rows):
        if i < len(gt["teams"]):
            exp_home, exp_away = gt["teams"][i]

            if _team_match(row.home_team, exp_home, canonical_map):
                result["teams_correct"] += 1
            else:
                result["errors"].append(f"Row {i+1} home: expected '{exp_home}', got '{row.home_team}'")

            if _team_match(row.away_team, exp_away, canonical_map):
                result["teams_correct"] += 1
            else:
                result["errors"].append(f"Row {i+1} away: expected '{exp_away}', got '{row.away_team}'")

            if row.home_team_mapped:
                result["teams_mapped"] += 1
            if row.away_team_mapped:
                result["teams_mapped"] += 1

        if row.streck_1 is not None and row.streck_x is not None and row.streck_2 is not None:
            result["streck_present"] += 1
        if row.odds_1 is not None and row.odds_x is not None and row.odds_2 is not None:
            result["odds_present"] += 1

        if row.issues:
            result["issues_summary"].extend(row.issues[:2])

    return result


def evaluate_legacy_result(extraction_result, ground_truth: dict, label: str) -> dict:
    """Evaluate legacy parser result against ground truth."""
    gt = ground_truth
    result = {
        "label": label,
        "expected_rows": gt["expected_rows"],
        "extracted_rows": extraction_result.total_rows,
        "complete_rows": extraction_result.complete_rows,
        "uncertain_rows": extraction_result.uncertain_rows,
        "incomplete_rows": extraction_result.incomplete_rows,
        "teams_correct": 0,
        "teams_total": gt["expected_rows"] * 2,
        "streck_present": 0,
        "odds_present": 0,
        "errors": [],
    }

    for i, row in enumerate(extraction_result.rows):
        if i < len(gt["teams"]):
            exp_home, exp_away = gt["teams"][i]

            if _team_match(row.home_team, exp_home):
                result["teams_correct"] += 1
            else:
                result["errors"].append(f"Row {i+1} home: expected '{exp_home}', got '{row.home_team}'")

            if _team_match(row.away_team, exp_away):
                result["teams_correct"] += 1
            else:
                result["errors"].append(f"Row {i+1} away: expected '{exp_away}', got '{row.away_team}'")

        if row.streck_1 is not None and row.streck_x is not None and row.streck_2 is not None:
            result["streck_present"] += 1
        if row.odds_1 is not None and row.odds_x is not None and row.odds_2 is not None:
            result["odds_present"] += 1

    return result


def run_comparison():
    """Run both pipelines on all test images and compare."""
    fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures", "coupons")

    all_results = []

    for filename, gt in GROUND_TRUTH.items():
        filepath = os.path.join(fixture_dir, filename)
        if not os.path.exists(filepath):
            print(f"SKIP: {filename} not found")
            continue

        with open(filepath, "rb") as f:
            image_bytes = f.read()

        print(f"\n{'='*70}")
        print(f"TESTING: {filename}")
        print(f"Expected: {gt['expected_rows']} rows")
        print(f"{'='*70}")

        # ---- New Pipeline ----
        print("\n--- NEW PIPELINE ---")
        t0 = time.time()
        try:
            pipeline_result = run_scanner_pipeline(image_bytes, filename)
            t_pipeline = time.time() - t0
            pipeline_eval = evaluate_pipeline_result(pipeline_result, gt, f"New Pipeline: {filename}")
            pipeline_eval["time_seconds"] = round(t_pipeline, 2)

            print(f"  Rows extracted: {pipeline_result.total_rows}/{gt['expected_rows']}")
            print(f"  OK/Uncertain/Failed: {pipeline_result.ok_rows}/{pipeline_result.uncertain_rows}/{pipeline_result.failed_rows}")
            print(f"  Teams correct: {pipeline_eval['teams_correct']}/{pipeline_eval['teams_total']}")
            print(f"  Teams mapped (canonical): {pipeline_eval['teams_mapped']}")
            print(f"  Streck complete rows: {pipeline_eval['streck_present']}")
            print(f"  Odds complete rows: {pipeline_eval['odds_present']}")
            print(f"  Time: {t_pipeline:.2f}s")
            print(f"  Preprocessing: {pipeline_result.preprocessing_applied}")

            if pipeline_eval["errors"]:
                print(f"  ERRORS:")
                for e in pipeline_eval["errors"][:5]:
                    print(f"    - {e}")

            # Show per-row details
            print(f"\n  Per-row details:")
            for i, row in enumerate(pipeline_result.rows):
                status_icon = {"ok": "OK", "uncertain": "?!", "failed": "XX"}.get(row.row_status, "??")
                mapped_h = f" (mapped from '{row.home_team_raw}')" if row.home_team_mapped and row.home_team != row.home_team_raw else ""
                mapped_a = f" (mapped from '{row.away_team_raw}')" if row.away_team_mapped and row.away_team != row.away_team_raw else ""
                streck = f"S:{row.streck_1}/{row.streck_x}/{row.streck_2}" if row.streck_1 is not None else "S:---"
                odds = f"O:{row.odds_1}/{row.odds_x}/{row.odds_2}" if row.odds_1 is not None else "O:---"
                print(f"    [{status_icon}] {row.home_team}{mapped_h} vs {row.away_team}{mapped_a} | {streck} | {odds} | conf={row.confidence_score:.2f}")
                if row.issues:
                    for issue in row.issues[:2]:
                        print(f"         ⚠ {issue}")

        except Exception as ex:
            print(f"  PIPELINE ERROR: {ex}")
            pipeline_eval = {"label": f"New Pipeline: {filename}", "error": str(ex)}
            t_pipeline = time.time() - t0

        # ---- Legacy Parser ----
        print("\n--- LEGACY PARSER ---")
        t0 = time.time()
        try:
            legacy_result = parse_coupon_image(image_bytes, filename, use_pipeline=False)
            t_legacy = time.time() - t0
            legacy_eval = evaluate_legacy_result(legacy_result, gt, f"Legacy: {filename}")
            legacy_eval["time_seconds"] = round(t_legacy, 2)

            print(f"  Rows extracted: {legacy_result.total_rows}/{gt['expected_rows']}")
            print(f"  Complete/Uncertain/Incomplete: {legacy_result.complete_rows}/{legacy_result.uncertain_rows}/{legacy_result.incomplete_rows}")
            print(f"  Teams correct: {legacy_eval['teams_correct']}/{legacy_eval['teams_total']}")
            print(f"  Streck complete rows: {legacy_eval['streck_present']}")
            print(f"  Odds complete rows: {legacy_eval['odds_present']}")
            print(f"  Time: {t_legacy:.2f}s")

            if legacy_eval["errors"]:
                print(f"  ERRORS:")
                for e in legacy_eval["errors"][:5]:
                    print(f"    - {e}")

            # Show per-row details
            print(f"\n  Per-row details:")
            for i, row in enumerate(legacy_result.rows):
                streck = f"S:{row.streck_1}/{row.streck_x}/{row.streck_2}" if row.streck_1 is not None else "S:---"
                odds = f"O:{row.odds_1}/{row.odds_x}/{row.odds_2}" if row.odds_1 is not None else "O:---"
                print(f"    [{row.confidence}] {row.home_team} vs {row.away_team} | {streck} | {odds}")

        except Exception as ex:
            print(f"  LEGACY ERROR: {ex}")
            legacy_eval = {"label": f"Legacy: {filename}", "error": str(ex)}
            t_legacy = time.time() - t0

        all_results.append({
            "filename": filename,
            "pipeline": pipeline_eval,
            "legacy": legacy_eval,
        })

    # ---- Summary ----
    print(f"\n\n{'='*70}")
    print("SUMMARY: NEW PIPELINE vs LEGACY")
    print(f"{'='*70}")

    pipeline_wins = 0
    legacy_wins = 0
    ties = 0

    for r in all_results:
        p = r.get("pipeline", {})
        l = r.get("legacy", {})

        if "error" in p or "error" in l:
            continue

        p_score = p.get("teams_correct", 0)
        l_score = l.get("teams_correct", 0)

        print(f"\n{r['filename']}:")
        print(f"  Pipeline: {p.get('extracted_rows', 0)} rows, {p_score} teams correct, {p.get('teams_mapped', 0)} mapped, {p.get('streck_present', 0)} streck, {p.get('odds_present', 0)} odds")
        print(f"  Legacy:   {l.get('extracted_rows', 0)} rows, {l_score} teams correct, {l.get('streck_present', 0)} streck, {l.get('odds_present', 0)} odds")

        if p_score > l_score:
            print(f"  WINNER: New Pipeline (+{p_score - l_score} teams)")
            pipeline_wins += 1
        elif l_score > p_score:
            print(f"  WINNER: Legacy (+{l_score - p_score} teams)")
            legacy_wins += 1
        else:
            # Tiebreaker: more extracted rows, streck, odds
            p_total = p.get("extracted_rows", 0) + p.get("streck_present", 0) + p.get("odds_present", 0)
            l_total = l.get("extracted_rows", 0) + l.get("streck_present", 0) + l.get("odds_present", 0)
            if p_total > l_total:
                print(f"  TIE on teams, Pipeline has more data ({p_total} vs {l_total})")
                pipeline_wins += 1
            elif l_total > p_total:
                print(f"  TIE on teams, Legacy has more data ({l_total} vs {p_total})")
                legacy_wins += 1
            else:
                print(f"  TIE")
                ties += 1

    print(f"\nOVERALL: Pipeline wins: {pipeline_wins}, Legacy wins: {legacy_wins}, Ties: {ties}")
    if pipeline_wins > legacy_wins:
        print("VERDICT: New pipeline is BETTER - safe to use as default")
    elif legacy_wins > pipeline_wins:
        print("VERDICT: Legacy is BETTER - keep legacy as default, pipeline behind flag")
    else:
        print("VERDICT: TIE - more testing needed to decide")

    # Save detailed results to JSON
    results_file = os.path.join(os.path.dirname(__file__), "e2e_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    run_comparison()
