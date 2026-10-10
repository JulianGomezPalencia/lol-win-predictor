-- The database structure. Each agent will add the tables it needs as we build it.
-- "IF NOT EXISTS" makes this safe to run every time the program starts.

-- Agent 1 (seeder): players sampled from every ranked tier.
CREATE TABLE IF NOT EXISTS players (
    puuid      TEXT PRIMARY KEY,   -- PRIMARY KEY = unique; the same player can't appear twice
    tier       TEXT NOT NULL,      -- IRON, BRONZE, ..., CHALLENGER
    division   TEXT NOT NULL,      -- I, II, III, IV (apex tiers are always I)
    lp         INTEGER NOT NULL,   -- league points
    seeded_at  TEXT NOT NULL       -- when we saw them, ISO format: 2026-10-09T09:00:00
);

-- Agent 2 (match discovery): the matches we want to predict. This table is the pipeline's
-- job queue: each agent picks up rows by `status` and moves them to the next one:
--   pending  --(Agent 3)-->  fetched  --(Agent 4)-->  history_done
--   (or `invalid` if Agent 3 finds a remake, etc.)
-- If anything crashes, re-running an agent simply continues with the rows still in its status.
CREATE TABLE IF NOT EXISTS target_matches (
    match_id       TEXT PRIMARY KEY,
    status         TEXT NOT NULL DEFAULT 'pending',
    seed_puuid     TEXT NOT NULL,   -- the player we found this match through
    seed_tier      TEXT NOT NULL,   -- their tier — lets us check the dataset stays balanced
    discovered_at  TEXT NOT NULL,
    -- Empty for now. Agent 3 fills these when it downloads the match:
    game_start_ts  INTEGER,         -- milliseconds since 1970 (Riot's format)
    patch          TEXT,            -- e.g. "16.8"
    blue_win       INTEGER          -- 1 = blue side won, 0 = red side won (SQLite has no true/false)
);
