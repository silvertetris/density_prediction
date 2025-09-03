import pandas as pd


def data_prex(path='../PopulationData', county='Songpa-gu'):
    immigrats_json = pd.read_json(path + '/countyImmigrate_data.json')
    born_json = pd.read_json(path + '/countyBorn_data.json')
    death_json = pd.read_json(path + '/countyDeath_data.json')
    population_json = pd.read_json(path + '/countyPopulation_data.json')

    immigrats_data = pd.DataFrame(immigrats_json)
    born_data = pd.DataFrame(born_json)
    death_data = pd.DataFrame(death_json)
    population_data = pd.DataFrame(population_json)

    born_data = born_data[born_data['C1_NM_ENG'] == county].reset_index(drop=True)
    death_data = death_data[death_data['C1_NM_ENG'] == county].reset_index(drop=True)
    immigrats_data = immigrats_data[immigrats_data['C1_NM_ENG'] == county].reset_index(drop=True)
    if county == 'Gangdong-gu': #강동구만 이상함
        population_data = population_data[population_data['C1_NM_ENG'] == 'Gang-dong'].reset_index(drop=True)
    else:
        population_data = population_data[population_data['C1_NM_ENG'] == county].reset_index(drop=True)

    immigrats_data = immigrats_data[immigrats_data['ITM_NM_ENG'] == "Netmigration(Administrative reports)"].reset_index(
        drop=True)
    population_data = population_data[population_data['ITM_NM_ENG'] == 'Koreans (Total)'].reset_index(drop=True)
    dfs = [immigrats_data, born_data, death_data, population_data]
    for i, df in enumerate(dfs):
        df['PRD_DE'] = df['PRD_DE'].astype(int)

    #max(min), min(max)
    starts = [df['PRD_DE'].min() for df in dfs]
    ends = [df['PRD_DE'].max() for df in dfs]

    min_start = max(starts)
    max_end = min(ends)
    for i, df in enumerate(dfs):
        dfs[i] = df[df['PRD_DE'].between(min_start, max_end)].reset_index(drop=True)

    immigrats_data, born_data, death_data, population_data = dfs

    for df in (immigrats_data, born_data, death_data, population_data):
        df['YM'] = pd.to_datetime(df['PRD_DE'].astype(str), format='%Y%m')  # yyyy-mm-dd 형식으로 바꿈

    idx = pd.date_range(
        start=pd.to_datetime(str(min_start), format='%Y%m'),
        end=pd.to_datetime(str(max_end), format='%Y%m'),
        freq='MS'
    )

    result = pd.DataFrame(index=idx)
    result['immigrants'] = immigrats_data.set_index('YM')['DT'].reindex(idx)
    result['born'] = born_data.set_index('YM')['DT'].reindex(idx)
    result['death'] = death_data.set_index('YM')['DT'].reindex(idx)
    result['population'] = population_data.set_index('YM')['DT'].reindex(idx)

    # 전체 출력 option
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    return result, min_start, max_end

