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

-- Agent 3 (match fetcher): the 10 players of each valid target match.
-- Agent 4 will read this table to know whose history to download.
CREATE TABLE IF NOT EXISTS match_participants (
    match_id       TEXT NOT NULL,
    puuid          TEXT NOT NULL,
    team_id        INTEGER NOT NULL,  -- 100 = blue side, 200 = red side
    team_position  TEXT NOT NULL,     -- TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
    champion_id    INTEGER NOT NULL,
    champion_name  TEXT NOT NULL,
    win            INTEGER NOT NULL,  -- 1 / 0
    PRIMARY KEY (match_id, puuid)     -- a player appears once per match (a "composite" key: two columns together)
);

-- Agent 4 (history builder): for each fetched target match, each player's last N ranked games
-- played BEFORE that match started. One row per (target match, player, history game).
-- We store a small summary instead of the whole JSON so feature engineering is a simple SQL query.
CREATE TABLE IF NOT EXISTS player_history (
    target_match_id   TEXT NOT NULL,
    puuid             TEXT NOT NULL,
    history_rank      INTEGER NOT NULL,  -- 0 = their most recent game before the target match
    history_match_id  TEXT NOT NULL,
    game_start_ts     INTEGER NOT NULL,  -- always < the target match's game_start_ts (no leakage)
    game_duration     INTEGER NOT NULL,  -- seconds
    remake            INTEGER NOT NULL,  -- 1 if that history game was a remake (feature code can skip it)
    win               INTEGER NOT NULL,
    champion_id       INTEGER NOT NULL,
    champion_name     TEXT NOT NULL,
    team_position     TEXT NOT NULL,     -- can be '' when Riot couldn't tell the lane
    kills             INTEGER NOT NULL,
    deaths            INTEGER NOT NULL,
    assists           INTEGER NOT NULL,
    PRIMARY KEY (target_match_id, puuid, history_rank)
);

-- Agent 5 (player snapshot): rank and champion mastery of every match participant, read TODAY.
-- These are snapshots, not values at match time — only trustworthy for recent matches
-- (see the leakage notes in NOTES.md). Rank columns are NULL for unranked players.
CREATE TABLE IF NOT EXISTS player_snapshots (
    puuid          TEXT PRIMARY KEY,
    tier           TEXT,               -- NULL = unranked in solo queue
    division       TEXT,
    lp             INTEGER,
    ranked_wins    INTEGER,
    ranked_losses  INTEGER,
    snapshot_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS champion_mastery (
    puuid             TEXT NOT NULL,
    champion_id       INTEGER NOT NULL,
    mastery_level     INTEGER NOT NULL,
    mastery_points    INTEGER NOT NULL,
    last_play_time    INTEGER NOT NULL,  -- milliseconds since 1970
    snapshot_at       TEXT NOT NULL,
    PRIMARY KEY (puuid, champion_id)
);
