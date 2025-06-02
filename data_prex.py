

path = './PopulationData'
def data_prex(path = './PopulationData'):
    import pandas as pd
    abroadmove_json = pd.read_json(path + '/kosis_abroadMove.json')
    abroadmove_data = pd.DataFrame(abroadmove_json)
    bornDeath_json = pd.read_json(path + '/kosis_bornDeath.json')
    bornDeath_data = pd.DataFrame(bornDeath_json)
    grdp_json = pd.read_json(path + '/kosis_grdp.json')
    grdp_data = pd.DataFrame(grdp_json)
    internalMove_json = pd.read_json(path + '/kosis_internalMove.json')
    internalMove_data = pd.DataFrame(internalMove_json)
    populationDensity_json = pd.read_json(path + '/kosis_populationDensity.json')
    populationDensity_data = pd.DataFrame(populationDensity_json)
    born_data = bornDeath_data[bornDeath_data['C2_NM_ENG'] == 'Live births(persons)'].reset_index(drop=True)
    death_data = bornDeath_data[bornDeath_data['C2_NM_ENG'] == 'Deaths(persons)'].reset_index(drop=True)
    grdp_data = grdp_data[grdp_data['ITM_NM'] == '실질'].reset_index(drop=True)
    print(populationDensity_data['PRD_DE'])
    print(populationDensity_data['DT'])
    print(grdp_data.head(5))

    import numpy as np
    import pandas as pd

    # 1. 연도별(y) 데이터 -> 월별(yyyymm)로 변환 함수
    def convert_yearly_to_monthly(df, date_col='PRD_DE', value_cols=[]):
        df[date_col] = df[date_col].astype(str)

        if df[date_col].str.len().max() == 4:
            # 연도별 데이터를 월별 템플릿으로 확장 (해당 연도의 1월에만 값 넣기)
            monthly_rows = []
            for _, row in df.iterrows():
                year = row[date_col]
                for m in range(1, 13):
                    month_str = str(m).zfill(2)
                    prd_de = year + month_str
                    data_row = {date_col: prd_de}
                    for col in value_cols:
                        data_row[col] = row[col] if m == 1 else np.nan
                    monthly_rows.append(data_row)
            df = pd.DataFrame(monthly_rows)

        for col in value_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    # 2. 기준 월별 데이터프레임 생성 (1990년 1월 ~ 2020년 12월, yyyymm 문자열)
    date_range = pd.date_range(start='1990-01-01', end='2020-12-01', freq='MS')  # 월 시작일 기준
    base_df = pd.DataFrame({'PRD_DE': date_range.strftime('%Y%m')})

    # 3. 데이터들 불러오기 및 전처리

    # (예: bornDeath_data -> 출생, 사망 분리)
    born_data = bornDeath_data[bornDeath_data['C2_NM_ENG'] == 'Live births(persons)'].reset_index(drop=True)
    death_data = bornDeath_data[bornDeath_data['C2_NM_ENG'] == 'Deaths(persons)'].reset_index(drop=True)

    born_data = convert_yearly_to_monthly(born_data, date_col='PRD_DE', value_cols=['DT'])
    death_data = convert_yearly_to_monthly(death_data, date_col='PRD_DE', value_cols=['DT'])

    # 자연증가 (출생 - 사망)
    nat_df = pd.merge(born_data[['PRD_DE', 'DT']], death_data[['PRD_DE', 'DT']], on='PRD_DE', how='outer',
                      suffixes=('_births', '_deaths'))
    nat_df['natural_increase'] = nat_df['DT_births'] - nat_df['DT_deaths']
    nat_df = nat_df[['PRD_DE', 'natural_increase']]

    # abroadmove_data
    abroad_df = convert_yearly_to_monthly(abroadmove_data, date_col='PRD_DE', value_cols=['DT'])
    abroad_df = abroad_df[['PRD_DE', 'DT']].rename(columns={'DT': 'abroad_move'})

    # internalMove_data
    internal_df = convert_yearly_to_monthly(internalMove_data, date_col='PRD_DE', value_cols=['DT'])
    internal_df = internal_df[['PRD_DE', 'DT']].rename(columns={'DT': 'internal_move'})

    grdp_df = convert_yearly_to_monthly(grdp_data, date_col='PRD_DE', value_cols=['DT'])
    grdp_df = grdp_df[['PRD_DE', 'DT']].rename(columns={'DT': 'grdp'})

    # populationDensity_data
    population_df = convert_yearly_to_monthly(populationDensity_data, date_col='PRD_DE', value_cols=['DT'])
    population_df = population_df[['PRD_DE', 'DT']].rename(columns={'DT': 'population_density'})

    # 4. base_df와 병합 & 결측치 보간 함수 (앞뒤 NaN도 채우도록 수정)
    def merge_and_interpolate(base, df, on='PRD_DE', value_col=None):
        merged = pd.merge(base, df[[on, value_col]], on=on, how='left')
        merged[value_col] = merged[value_col].interpolate(method='linear')  # 선형 보간
        merged[value_col] = merged[value_col].bfill()  # 앞쪽 NaN을 뒤쪽 값으로 채움
        merged[value_col] = merged[value_col].ffill()  # 뒤쪽 NaN을 앞쪽 값으로 채움
        return merged

    # 5. 각 데이터 병합 및 보간
    nat_df = merge_and_interpolate(base_df, nat_df, value_col='natural_increase')
    abroad_df = merge_and_interpolate(base_df, abroad_df, value_col='abroad_move')
    internal_df = merge_and_interpolate(base_df, internal_df, value_col='internal_move')
    grdp_df = merge_and_interpolate(base_df, grdp_df, value_col='grdp')
    population_df = merge_and_interpolate(base_df, population_df, value_col='population_density')

    # 6. 최종 데이터프레임 병합
    final_df = base_df.copy()
    final_df = final_df.merge(nat_df[['PRD_DE', 'natural_increase']], on='PRD_DE')
    final_df = final_df.merge(abroad_df[['PRD_DE', 'abroad_move']], on='PRD_DE')
    final_df = final_df.merge(internal_df[['PRD_DE', 'internal_move']], on='PRD_DE')
    final_df = final_df.merge(grdp_df[['PRD_DE', 'grdp']], on='PRD_DE')
    final_df = final_df.merge(population_df[['PRD_DE', 'population_density']], on='PRD_DE')

    # 7. 결과 확인
    print(final_df.head(100))
    return final_df


data_prex(path)
