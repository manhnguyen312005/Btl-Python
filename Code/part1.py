import random
import time
import pandas as pd
from bs4 import BeautifulSoup, Comment
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


options = Options()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36")

browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
browser.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
    "source": """
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined})
        """
})


stat_sources = [
    ("https://fbref.com/en/comps/9/stats/Premier-League-Stats#all_stats_standard", "all_stats_standard", "D:/btl/player_stats_standard.csv"),
    ("https://fbref.com/en/comps/9/keepers/Premier-League-Stats#all_stats_keeper_saves", "all_stats_keeper", "D:/btl/player_goalkeeping.csv"),
    ("https://fbref.com/en/comps/9/shooting/Premier-League-Stats#all_stats_shooting", "all_stats_shooting", "D:/btl/player_shooting.csv"),
    ("https://fbref.com/en/comps/9/passing/Premier-League-Stats#all_stats_passing", "all_stats_passing", "D:/btl/player_passing.csv"),
    ("https://fbref.com/en/comps/9/gca/Premier-League-Stats#all_stats_gca", "all_stats_gca", "D:/btl/player_gca.csv"),
    ("https://fbref.com/en/comps/9/defense/Premier-League-Stats#all_stats_defense", "all_stats_defense", "D:/btl/player_defense.csv"),
    ("https://fbref.com/en/comps/9/possession/Premier-League-Stats#all_stats_possession", "all_stats_possession", "D:/btl/player_possession.csv"),
    ("https://fbref.com/en/comps/9/misc/Premier-League-Stats#all_stats_misc", "all_stats_misc", "D:/btl/player_misc.csv")
]


for url, div_id, file_path in stat_sources:
        browser.get(url)
        time.sleep(random.uniform(3.5, 7.5))
        soup = BeautifulSoup(browser.page_source, "html.parser")
        div = soup.find("div", id=div_id)
        comment = div.find(string=lambda text: isinstance(text, Comment)) if div else None
        table_html = BeautifulSoup(comment, "html.parser").find("table") if comment else div.find("table") if div else None
        if table_html:
            df = pd.read_html(StringIO(str(table_html)))[0]
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
            print(f"\u2714\ufe0f Lưu: {file_path}")
        else:
            print(f"\u274c Không tìm thấy bảng: {file_path}")

browser.quit()


for path in [item[2] for item in stat_sources]:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines[1:])


def combine_stat_table(main_df, filepath, selected_cols):
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
            return pd.merge(main_df, df[["Player", "Squad"] + selected_cols], on=["Player", "Squad"], how="left")
        except:
            return main_df


data = pd.read_csv("D:/btl/player_stats_standard.csv")
data.columns = data.columns.str.strip()
data = data[pd.to_numeric(data["Min"], errors="coerce") > 90].copy()
data["First Name"] = data["Player"].apply(lambda x: x.split()[0])
data.sort_values("First Name", inplace=True)

columns_to_keep = ["Player", "Nation", "Squad", "Pos", "Age", "MP", "Starts", "Min",
                    "Gls", "Ast", "CrdY", "CrdR", "xG", "xAG", "PrgC", "PrgP", "PrgR",
                    "Gls.1", "Ast.1", "xG.1", "xAG.1"]
data = data[columns_to_keep]


data = combine_stat_table(data, "D:/btl/player_goalkeeping.csv", ["GA90", "Save%", "CS%"])
data = combine_stat_table(data, "D:/btl/player_shooting.csv", ["SoT%", "SoT/90", "G/Sh", "Dist"])
data = combine_stat_table(data, "D:/btl/player_passing.csv", ["Cmp", "Cmp%", "TotDist", "Cmp%.1", "Cmp%.2", "Cmp%.3", "KP", "1/3", "PPA", "CrsPA", "PrgP"])
data = combine_stat_table(data, "D:/btl/player_gca.csv", ["SCA", "SCA90", "GCA", "GCA90"])
data = combine_stat_table(data, "D:/btl/player_defense.csv", ["Tkl", "TklW", "Att", "Lost", "Blocks", "Sh", "Pass", "Int"])
data = combine_stat_table(data, "D:/btl/player_possession.csv", ["Touches", "Def Pen", "Def 3rd", "Mid 3rd", "Att 3rd", "Att Pen", 
                                                                 "Att", "Succ%", "Tkld%", "Carries", "PrgDist", "PrgC", "1/3", "CPA", "Mis", "Dis", "Rec", "PrgR"])
data = combine_stat_table(data, "D:/btl/player_misc.csv", ["Fls", "Fld", "Off", "Crs", "Recov", "Won", "Lost", "Won%"])


data.fillna("N/a", inplace=True)
data.to_csv("D:/btl/results.csv", index=False, encoding="utf-8-sig")
print("\n✔\ufe0f Hoàn thành! File: D:/btl/results.csv")