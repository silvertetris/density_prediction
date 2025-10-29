import pandas as pd

def data_prex(path = '../PopulationData/townScale/'):
    population_json1 = pd.read_json(path+'seoul_town_scale_data_1.json', dtype={'C1': str})
    population_json2 = pd.read_json(path+'seoul_town_scale_data_2.json', dtype={'C1': str})
    population_json3 = pd.read_json(path+'seoul_town_scale_data_3.json', dtype={'C1': str})

    population_json = pd.concat([population_json1, population_json2, population_json3])
    population_data = pd.DataFrame(population_json)

    df = (
        population_data.loc[:, ["PRD_DE", "C1", "DT"]]
        .rename(columns={"PRD_DE": "date", "C1": "location_code", "DT": "value"})
        .dropna(subset=["date", "location_code", "value"])
    )
    df["location_code"] = df["location_code"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m", errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    print("[DEBUG] unique code lengths:", df["location_code"].str.len().value_counts().to_dict())
    print("[DEBUG] sample codes:", df["location_code"].unique()[:10])
    return df


def town_wide_data(df: pd.DataFrame): #wide 시계열 용 데이터
    tmp = df.copy()
    tmp["code"] = tmp["location_code"].astype(str).str.strip()
    dong = tmp[tmp["code"].str.len() == 10].copy()
    wide = dong.pivot(index="date", columns="code", values="value").sort_index()
    wide = wide.apply(pd.to_numeric, errors="coerce")
    return wide

data = town_wide_data(data_prex())
print(data.shape, data.last_valid_index())