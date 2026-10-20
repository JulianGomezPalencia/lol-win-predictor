"""
Run the whole data-collection pipeline, agent by agent.

    python -m scripts.run_pipeline                       # every step
    python -m scripts.run_pipeline --steps fetch,history # only some steps
    python -m scripts.run_pipeline --limit 20            # cap the slow steps (history, snapshot)

Every agent is safe to re-run: each one only picks up the work that's still left (the
`status` column of target_matches, or players without a snapshot). So if anything stops
the run — Ctrl+C, a crash, the laptop sleeping — just run the same command again.

Steps, in order:
  seed      Agent 1  sample players from every tier (~31 calls)
  discover  Agent 2  find their recent ranked matches   (~1 call per player)
  fetch     Agent 3  download + validate those matches  (~1 call per match)
  history   Agent 4  each player's games before each match (slowest: up to ~110 calls per match)
  snapshot  Agent 5  rank + champion mastery of participants (~2 calls per player)
"""
import argparse

from src.agents import history_builder, match_discovery, match_fetcher, player_snapshot, seeder
from src.storage import db

STEPS = ["seed", "discover", "fetch", "history", "snapshot"]


def main():
    parser = argparse.ArgumentParser(description="Run the LoL data-collection pipeline.")
    parser.add_argument("--steps", default=",".join(STEPS),
                        help=f"comma-separated steps to run, in pipeline order (default: {','.join(STEPS)})")
    parser.add_argument("--limit", type=int, help="max target matches (history) / players (snapshot) this run")
    args = parser.parse_args()

    steps = [step.strip() for step in args.steps.split(",")]
    unknown = [step for step in steps if step not in STEPS]
    if unknown:
        parser.error(f"unknown step(s): {', '.join(unknown)}. Choose from: {', '.join(STEPS)}")

    conn = db.connect()
    try:
        for step in STEPS:  # always pipeline order, whatever order they were typed in
            if step not in steps:
                continue
            print(f"\n=== {step} ===")
            if step == "seed":
                print(f"Saved {seeder.seed(conn)} players.")
            elif step == "discover":
                print(f"{match_discovery.discover(conn)} new matches.")
            elif step == "fetch":
                print("Fetched:", match_fetcher.fetch_pending(conn))
            elif step == "history":
                print(f"{history_builder.build_histories(conn, limit=args.limit)} matches completed.")
            elif step == "snapshot":
                print(f"{player_snapshot.snapshot_players(conn, limit=args.limit)} players snapshotted.")
        print("\nQueue:", db.count_target_matches_by_status(conn))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
