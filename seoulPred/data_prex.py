import pandas as pd
import numpy as np


def data_prex(path='./PopulationData'):
    # 1) 원본 JSON 불러오기
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

    # 2) 불필요 행 필터링
    born_data = bornDeath_data[bornDeath_data['C2_NM_ENG'] == 'Live births(persons)'].reset_index(drop=True)
    death_data = bornDeath_data[bornDeath_data['C2_NM_ENG'] == 'Deaths(persons)'].reset_index(drop=True)
    grdp_data = grdp_data[grdp_data['ITM_NM'] == '실질'].reset_index(drop=True)

    def convert_yearly_to_monthly(df, date_col='PRD_DE', value_cols=[]):
        df = df.copy()
        df[date_col] = df[date_col].astype(str)

        # 연도 문자열(길이 4)인 경우만 확장
        if df[date_col].str.len().max() == 4:
            rows = []
            for _, row in df.iterrows():
                year = row[date_col]
                for m in range(1, 13):
                    prd_de = year + str(m).zfill(2)
                    new_row = {date_col: prd_de}
                    # 모든 월에 동일한 값 할당
                    for col in value_cols:
                        new_row[col] = row[col]
                    rows.append(new_row)
            df = pd.DataFrame(rows)

        # 숫자형 변환
        for col in value_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    # 4) 기준 월 프레임 생성 (1990.01 ~ 2020.12)
    date_range = pd.date_range(start='1990-01-01', end='2020-12-01', freq='MS')
    base_df = pd.DataFrame({'PRD_DE': date_range.strftime('%Y%m')})

    # 5) 각 데이터프레임 처리
    born_df = convert_yearly_to_monthly(born_data, date_col='PRD_DE', value_cols=['DT'])
    death_df = convert_yearly_to_monthly(death_data, date_col='PRD_DE', value_cols=['DT'])
    abroad_df = convert_yearly_to_monthly(abroadmove_data, date_col='PRD_DE', value_cols=['DT'])
    internal_df = convert_yearly_to_monthly(internalMove_data, date_col='PRD_DE', value_cols=['DT'])
    grdp_df = convert_yearly_to_monthly(grdp_data, date_col='PRD_DE', value_cols=['DT'])
    pop_df = convert_yearly_to_monthly(populationDensity_data, date_col='PRD_DE', value_cols=['DT'])

    # 6) 자연증가 계산 (출생−사망)
    nat = pd.merge(
        born_df[['PRD_DE', 'DT']].rename(columns={'DT': 'births'}),
        death_df[['PRD_DE', 'DT']].rename(columns={'DT': 'deaths'}),
        on='PRD_DE', how='outer'
    )
    nat['natural_increase'] = nat['births'] - nat['deaths']
    nat_df = nat[['PRD_DE', 'natural_increase']]

    # 7) 컬럼 이름 통일
    abroad_df = abroad_df[['PRD_DE', 'DT']].rename(columns={'DT': 'abroad_move'})
    internal_df = internal_df[['PRD_DE', 'DT']].rename(columns={'DT': 'internal_move'})
    grdp_df = grdp_df[['PRD_DE', 'DT']].rename(columns={'DT': 'grdp'})
    pop_df = pop_df[['PRD_DE', 'DT']].rename(columns={'DT': 'population_density'})

    # 8) 병합 & 계단식 채우기 함수
    def merge_and_fill(base, df, on='PRD_DE', value_col=None):
        m = pd.merge(base, df[[on, value_col]], on=on, how='left')
        m[value_col] = m[value_col].bfill()  # 앞쪽 NaN 채우기
        m[value_col] = m[value_col].ffill()  # 뒤쪽 NaN 채우기
        return m

    nat_df = merge_and_fill(base_df, nat_df, value_col='natural_increase')
    abroad_df = merge_and_fill(base_df, abroad_df, value_col='abroad_move')
    internal_df = merge_and_fill(base_df, internal_df, value_col='internal_move')
    grdp_df = merge_and_fill(base_df, grdp_df, value_col='grdp')
    population_df = merge_and_fill(base_df, pop_df, value_col='population_density')

    # 9) 최종 합치기
    final_df = base_df.copy()
    final_df = final_df.merge(nat_df[['PRD_DE', 'natural_increase']], on='PRD_DE')
    final_df = final_df.merge(abroad_df[['PRD_DE', 'abroad_move']], on='PRD_DE')
    final_df = final_df.merge(internal_df[['PRD_DE', 'internal_move']], on='PRD_DE')
    final_df = final_df.merge(grdp_df[['PRD_DE', 'grdp']], on='PRD_DE')
    final_df = final_df.merge(population_df[['PRD_DE', 'population_density']], on='PRD_DE')

    # 10) 결과 확인
    print(final_df.head(12))
    return final_df


# 실행 예
if __name__ == '__main__':
    df = data_prex('../PopulationData')
