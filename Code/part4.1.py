import requests
import pandas as pd

url = "https://www.footballtransfers.com/us/values/actions/most-valuable-football-players/overview"
headers = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "vi,vi-VN;q=0.9,en-US;q=0.8,en;q=0.7",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Cookie": "FootballTransfers_Language=4; _cb=D7FNq7C1N35-C-0Vam; usprivacy=1N--; _gid=GA1.2.1717814782.1746813889; _cb_svref=external; _gat_gtag_UA_177878282_1=1; _chartbeat5=; _ga=GA1.2.2066071459.1746203180; _ga_6D1E4VM1TV=GS2.1.1746818214$s0c6$g1$t1746818249$j25$l0$h0; _ga_9GLV86Z159=GS2.1.1746818214$s0c6$g1$t1746818249$j0$l0$h0; _chartbeat2=.1746203179023.1746818249917.110000001.DJcYR9DltAiDc00fPDsKUrGC7Dlgk.6",
    "Origin": "https://www.footballtransfers.com",
    "Referer": "https://www.footballtransfers.com/us/values/players/most-valuable-soccer-players/playing-in-uk-premier-league",
    "Sec-Ch-Ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest"
}

payload = {
    "orderBy": "estimated_value",
    "orderByDescending": 1,
    "page": 1,
    "pages": 0,
    "pageItems": 25,
    "positionGroupId": "all",
    "mainPositionId": "all",
    "playerRoleId": "all",
    "age": "all",
    "countryId": "all",
    "tournamentId": 31
}

all_data = []  


for i in range(1, 23):  
    payload["page"] = i
    response = requests.post(url, headers=headers, data=payload)

    if response.status_code == 200:
        data = response.json()
        records = data["records"]
        df = pd.DataFrame(records)
        
       
        df_filtered = df[["player_name", "estimated_value"]]
        
       
        all_data.append(df_filtered)
       
    else:
        print("Error:", response.status_code)


final_df = pd.concat(all_data, ignore_index=True)



playtime = pd.read_csv('results.csv')
df_play_time_filtered = playtime[playtime["Min"] > 900]
df_play_time_filtered.rename(columns={"Player": "player_name"}, inplace=True)
df_combined = pd.merge(df_play_time_filtered, final_df, on="player_name",how="left")
df_no_value = df_combined[df_combined["estimated_value"].isna()]



import time

import requests
import time
from urllib.parse import urlencode
import unidecode  


for index, row in df_combined[df_combined["estimated_value"].isna()].iterrows():
    player_name = row["player_name"]


    safe_player_name = unidecode.unidecode(player_name)

    url = "https://www.footballtransfers.com/us/search/actions/search"

    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "vi,vi-VN;q=0.9,en-US;q=0.8,en;q=0.7",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://www.footballtransfers.com",
        "Referer": f"https://www.footballtransfers.com/us/search?search_value={safe_player_name}",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }

    payload = {
        "search_page": 1,
        "search_value": safe_player_name,
        "players": 1,
        "teams": 1
    }

    try:
        response = requests.post(url, headers=headers, data=payload)

        if response.status_code == 200:
            json_data = response.json()

            if json_data.get("hits"):
                first_hit = json_data["hits"][0]["document"]
                player_value = first_hit.get("transfer_value")

                if player_value:
                    df_combined.at[index, "estimated_value"] = player_value
                    print(f"✅ Đã cập nhật {player_name} với giá {player_value}")
                else:
                    print(f"⚠️ Không có giá cho {player_name}")
            else:
                print(f"❌ Không tìm thấy {player_name}")
        else:
            print(f"🔁 Lỗi khi gọi API cho {player_name}: {response.status_code}")
    except Exception as e:
        print(f"🚨 Lỗi với {player_name}: {e}")
df_combined.to_csv("df_combined.csv", index=False, encoding="utf-8-sig")