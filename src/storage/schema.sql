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
