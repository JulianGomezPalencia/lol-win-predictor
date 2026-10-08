# LoL Win Predictor

Software Engineering project at Universidad Industrial de Santander (UIS).

Predicts the probability that a League of Legends ranked match will be won **before it starts**, using
each player's recent history: their last 10 results, whether they're in their usual lane, and whether
they're playing a champion they're comfortable on.

## Setup

```bash
python3 -m venv .venv            # create an isolated Python environment
source .venv/bin/activate        # activate it (do this every time you open a new terminal)
pip install -r requirements.txt  # install the libraries
cp .env.example .env             # then paste your Riot API key into .env
```

## Project layout

| Folder | What goes there |
|---|---|
| `src/riot_client/` | The only code that talks to the Riot API |
| `src/agents/` | (coming) Pipeline stages that collect data |
| `scripts/` | Small runnable scripts |
| `tests/` | Automated tests |
| `data/` | Downloaded matches and the database (not in git) |
| `notebooks/` | Data exploration |

## Roadmap

- [ ] Exercise 1: first API calls (`scripts/hello_riot.py`)
- [ ] Shared Riot client with rate limiting and caching
- [ ] Database + data collection agents
- [ ] Feature engineering
- [ ] Model training

---
This project isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone
officially involved in producing or managing Riot Games properties. Riot Games and all associated properties
are trademarks or registered trademarks of Riot Games, Inc.
