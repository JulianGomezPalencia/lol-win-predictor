import sys
from src.riot_client.endpoints import get_match, get_match_ids, get_puuid


def main():
    if len(sys.argv) != 2 or "#" not in sys.argv[1]:
        print('Usage: python -m scripts.hello_riot "Name#TAG"')
        sys.exit(1)

    game_name, tag_line = sys.argv[1].split("#", 1)

    puuid = get_puuid(game_name, tag_line)
    print("PUUID:", puuid)

    match_ids = get_match_ids(puuid)
    print("Last ranked games:", match_ids)

    if not match_ids:
        print("No matches found")
        return
    
    match = get_match(match_ids[0])
    match_info = match["info"]

    for player in match_info["participants"]:
        if player["puuid"] == puuid:
            win = player["win"]
            champion_name = player["championName"]
            team_position = player["teamPosition"]
            print(f"Win: {win}")
            print(f"Champion name: {champion_name}")
            print(f"Team position: {team_position}")
            break

    game_duration = match_info["gameDuration"]
    game_version = match_info["gameVersion"]
    print(f"Game duration: {game_duration // 60}:{game_duration % 60:02d}")
    print(f"Game version: {game_version}")

    




if __name__ == "__main__":
    main()
