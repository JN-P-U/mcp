# 재무제표 분석(dart_api)

import json
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests


def register_api_key(api_key):
    """
    DART API 키를 등록합니다.
    """
    config_dir = os.path.expanduser("~/.mcp")
    config_file = os.path.join(config_dir, "dart_config.json")

    # 설정 디렉토리가 없으면 생성
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # 설정 파일이 없으면 생성
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            json.dump({"api_key": api_key}, f)
    else:
        # 기존 설정 파일 읽기
        with open(config_file, "r") as f:
            config = json.load(f)

        # API 키 업데이트
        config["api_key"] = api_key

        # 설정 파일 저장
        with open(config_file, "w") as f:
            json.dump(config, f)

    return True


def get_api_key():
    """
    등록된 DART API 키를 가져옵니다.
    """
    # config_file = os.path.join(os.path.expanduser("~/.mcp"), "dart_config.json")

    # if not os.path.exists(config_file):
    #     return None

    # with open(config_file, "r") as f:
    #     config = json.load(f)

    # return config.get("api_key")
    api_key = os.getenv("DART_API_KEY")
    return api_key


def fetch_corp_codes(api_key):
    """
    DART API를 통해 상장기업 목록과 종목코드를 가져옵니다.
    """
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    params = {
        "crtfc_key": api_key,
    }

    response = requests.get(url, params=params)
    print(f"API 응답 상태 코드: {response.status_code}")

    if response.status_code != 200:
        print("상장기업 코드를 가져오는데 실패했습니다.")
        return pd.DataFrame()

    # 응답이 ZIP 파일인 경우 처리
    try:
        # ZIP 파일을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        # ZIP 파일 압축 해제
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(temp_file_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # XML 파일 경로
            xml_file_path = os.path.join(temp_dir, "CORPCODE.xml")

            # XML 파일이 존재하는지 확인
            if not os.path.exists(xml_file_path):
                print("XML 파일을 찾을 수 없습니다.")
                os.unlink(temp_file_path)
                return pd.DataFrame()

            # XML 파싱
            try:
                tree = ET.parse(xml_file_path)
                root = tree.getroot()

                # 결과 저장
                result = []

                # 각 기업 정보 추출
                for company in root.findall(".//list"):
                    corp_code = company.find("corp_code")
                    corp_name = company.find("corp_name")
                    stock_code = company.find("stock_code")
                    market = company.find("market")

                    if (
                        corp_code is not None
                        and corp_code.text
                        and corp_name is not None
                        and corp_name.text
                        and stock_code is not None
                        and stock_code.text
                        and stock_code.text.strip()
                    ):
                        result.append(
                            {
                                "corp_code": corp_code.text,
                                "corp_name": corp_name.text,
                                "stock_code": stock_code.text,
                                "market": market.text if market is not None else "",
                            }
                        )

                # 임시 파일 삭제
                os.unlink(temp_file_path)
                return pd.DataFrame(result)

            except ET.ParseError as e:
                print(f"XML 파싱 오류: {e}")
                os.unlink(temp_file_path)
                return pd.DataFrame()

    except Exception as e:
        print(f"데이터 처리 중 오류 발생: {e}")
        return pd.DataFrame()


def find_corp_code_by_name(companies_df, corp_name):
    """
    기업명으로 기업 코드를 찾습니다.
    """
    # 정확한 일치 검색
    exact_match = companies_df[companies_df["corp_name"] == corp_name]
    if not exact_match.empty:
        return exact_match.iloc[0]["corp_code"]

    # 부분 일치 검색
    partial_match = companies_df[
        companies_df["corp_name"].str.contains(corp_name, na=False)
    ]
    if not partial_match.empty:
        return partial_match.iloc[0]["corp_code"]

    return None


def find_corp_code_by_stock_code(companies_df, stock_code):
    """
    종목 코드로 기업 코드를 찾습니다.
    """
    # 종목 코드 정규화 (6자리 숫자로 변환)
    stock_code = stock_code.strip().zfill(6)

    # 종목 코드로 검색
    match = companies_df[companies_df["stock_code"] == stock_code]
    if not match.empty:
        return match.iloc[0]["corp_code"]

    # 종목 코드가 없는 경우 (빈 문자열 또는 '0'으로 채워진 경우)
    if stock_code == "000000":
        print("유효하지 않은 종목 코드입니다.")
        return None

    # 종목 코드가 6자리가 아닌 경우
    if len(stock_code) != 6:
        print("종목 코드는 6자리 숫자여야 합니다.")
        return None

    print(f"종목 코드 '{stock_code}'에 해당하는 기업을 찾을 수 없습니다.")
    return None


def fetch_financial_statements(api_key, corp_code, year):
    """
    특정 연도의 재무제표 데이터를 가져옵니다.
    """
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json"
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bsns_year": year,
        "reprt_code": "11011",  # 사업보고서
        "fs_div": "OFS",  # 재무제표 구분: O(별도), C(연결)
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if data.get("status") == "000":  # 정상 응답
            result = data.get("list", [])
            if not result:
                print(f"{year}년 데이터가 없습니다.")
            return result
        else:
            print(f"Error: {data.get('message')}")
            return []
    except Exception as e:
        print(f"재무제표 데이터를 가져오는 중 오류 발생: {e}")
        return []


def analyze_financial_statements(stock_code):
    """
    주식 코드를 기반으로 재무제표를 분석합니다.
    """
    # DART API 키 가져오기
    dart_api_key = get_api_key()
    if not dart_api_key:
        print("재무제표 데이터가 없거나 잘못된 형식입니다.")
        return {
            "income_statement": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "cash_flow": pd.DataFrame(),
            "growth_rates": pd.DataFrame(),
            "financial_health": {
                "부채비율 상태": "N/A",
                "유동비율 상태": "N/A",
                "영업이익률 상태": "N/A",
                "종합 평가": "N/A",
            },
        }

    # 상장기업 목록 가져오기
    print("상장기업 목록을 가져오는 중...")
    companies_df = fetch_corp_codes(dart_api_key)
    if companies_df is None:
        print("재무제표 데이터가 없거나 잘못된 형식입니다.")
        return None

    # 기업 정보 찾기
    company_info = companies_df[companies_df["stock_code"] == stock_code]
    if len(company_info) == 0:
        print("해당 종목 코드를 찾을 수 없습니다.")
        return None

    company_name = company_info.iloc[0]["corp_name"]
    corp_code = company_info.iloc[0]["corp_code"]

    print(f"\n선택한 기업: {company_name} ({stock_code})")
    print(f"기업 코드: {corp_code}")

    # 재무제표 데이터 저장용 딕셔너리
    financial_data = {
        "매출액": [],
        "영업이익": [],
        "당기순이익": [],
        "자산총계": [],
        "부채총계": [],
        "자본총계": [],
        "유동자산": [],
        "유동부채": [],
        "영업활동현금흐름": [],
        "투자활동현금흐름": [],
        "재무활동현금흐름": [],
        "연도": [],
    }

    # 최근 5년간의 재무제표 데이터 가져오기
    current_year = datetime.now().year
    for year in range(current_year - 4, current_year + 1):
        print(f"{year}년 재무제표 데이터를 가져오는 중...")
        data = fetch_financial_statements(dart_api_key, corp_code, str(year))

        if data:
            print(f"{year}년 데이터 항목 수: {len(data)}")
            for i, item in enumerate(data[:3], 1):
                print(f"  항목 {i}: {item['account_nm']} = {item['thstrm_amount']}")

            # 데이터 추출
            found_items = set()  # 이미 찾은 항목을 추적
            for item in data:
                account_name = item["account_nm"]
                amount = int(item["thstrm_amount"].replace(",", ""))

                # 각 항목별로 한 번만 처리
                if "매출액" in account_name and "매출액" not in found_items:
                    print(f"찾은 항목: 매출액, 값: {amount}")
                    financial_data["매출액"].append(amount)
                    found_items.add("매출액")
                elif "영업이익" in account_name and "영업이익" not in found_items:
                    print(f"찾은 항목: 영업이익, 값: {amount}")
                    financial_data["영업이익"].append(amount)
                    found_items.add("영업이익")
                elif "당기순이익" in account_name and "당기순이익" not in found_items:
                    print(f"찾은 항목: 당기순이익, 값: {amount}")
                    financial_data["당기순이익"].append(amount)
                    found_items.add("당기순이익")
                elif "자산총계" in account_name and "자산총계" not in found_items:
                    print(f"찾은 항목: 자산총계, 값: {amount}")
                    financial_data["자산총계"].append(amount)
                    found_items.add("자산총계")
                elif "부채총계" in account_name and "부채총계" not in found_items:
                    print(f"찾은 항목: 부채총계, 값: {amount}")
                    financial_data["부채총계"].append(amount)
                    found_items.add("부채총계")
                elif "자본총계" in account_name and "자본총계" not in found_items:
                    print(f"찾은 항목: 자본총계, 값: {amount}")
                    financial_data["자본총계"].append(amount)
                    found_items.add("자본총계")
                elif "유동자산" in account_name and "유동자산" not in found_items:
                    print(f"찾은 항목: 유동자산, 값: {amount}")
                    financial_data["유동자산"].append(amount)
                    found_items.add("유동자산")
                elif "유동부채" in account_name and "유동부채" not in found_items:
                    print(f"찾은 항목: 유동부채, 값: {amount}")
                    financial_data["유동부채"].append(amount)
                    found_items.add("유동부채")
                elif (
                    "영업활동현금흐름" in account_name
                    and "영업활동현금흐름" not in found_items
                ):
                    print(f"찾은 항목: 영업활동현금흐름, 값: {amount}")
                    financial_data["영업활동현금흐름"].append(amount)
                    found_items.add("영업활동현금흐름")
                elif (
                    "투자활동현금흐름" in account_name
                    and "투자활동현금흐름" not in found_items
                ):
                    print(f"찾은 항목: 투자활동현금흐름, 값: {amount}")
                    financial_data["투자활동현금흐름"].append(amount)
                    found_items.add("투자활동현금흐름")
                elif (
                    "재무활동현금흐름" in account_name
                    and "재무활동현금흐름" not in found_items
                ):
                    print(f"찾은 항목: 재무활동현금흐름, 값: {amount}")
                    financial_data["재무활동현금흐름"].append(amount)
                    found_items.add("재무활동현금흐름")

            # 찾지 못한 항목 출력
            for key in financial_data.keys():
                if key != "연도" and key not in found_items:
                    print(f"찾지 못한 항목: {key} - 데이터를 찾지 못했습니다.")
                    financial_data[key].append(np.nan)  # 찾지 못한 항목은 NaN으로 설정

            financial_data["연도"].append(year)
        else:
            print(f"Error: 조회된 데이타가 없습니다.")
            print("재무제표 데이터가 없거나 잘못된 형식입니다.")
            print(f"{year}년 데이터 분석 실패")

    # 데이터프레임 생성
    df = pd.DataFrame(financial_data)
    if df.empty:
        print("재무제표 데이터가 없거나 잘못된 형식입니다.")
        return None

    # 연도를 인덱스로 설정
    df.set_index("연도", inplace=True)

    # 손익계산서 데이터프레임
    income_statement = df[["매출액", "영업이익", "당기순이익"]].copy()
    income_statement["영업이익률"] = (
        income_statement["영업이익"] / income_statement["매출액"]
    ) * 100

    # 대차대조표 데이터프레임
    balance_sheet = df[
        ["자산총계", "부채총계", "자본총계", "유동자산", "유동부채"]
    ].copy()
    balance_sheet["부채비율"] = (
        balance_sheet["부채총계"] / balance_sheet["자본총계"]
    ) * 100
    balance_sheet["유동비율"] = (
        balance_sheet["유동자산"] / balance_sheet["유동부채"]
    ) * 100

    # 현금흐름표 데이터프레임
    cash_flow = df[["영업활동현금흐름", "투자활동현금흐름", "재무활동현금흐름"]].copy()

    # 성장률 계산
    growth_rates = pd.DataFrame(index=df.index)
    for col in ["매출액", "영업이익", "당기순이익"]:
        growth_rates[f"{col} 성장률"] = df[col].pct_change() * 100

    # 재무건전성 평가
    latest_year = df.index.max()
    latest_data = {
        "부채비율": balance_sheet.loc[latest_year, "부채비율"],
        "유동비율": balance_sheet.loc[latest_year, "유동비율"],
        "영업이익률": income_statement.loc[latest_year, "영업이익률"],
    }

    financial_health = {}

    # 부채비율 평가 (낮을수록 좋음, 200% 이하가 적정)
    if latest_data["부채비율"] <= 100:
        financial_health["부채비율 상태"] = "매우 좋음"
    elif latest_data["부채비율"] <= 200:
        financial_health["부채비율 상태"] = "좋음"
    elif latest_data["부채비율"] <= 300:
        financial_health["부채비율 상태"] = "주의"
    else:
        financial_health["부채비율 상태"] = "위험"

    # 유동비율 평가 (높을수록 좋음, 150% 이상이 적정)
    if latest_data["유동비율"] >= 200:
        financial_health["유동비율 상태"] = "매우 좋음"
    elif latest_data["유동비율"] >= 150:
        financial_health["유동비율 상태"] = "좋음"
    elif latest_data["유동비율"] >= 100:
        financial_health["유동비율 상태"] = "주의"
    else:
        financial_health["유동비율 상태"] = "위험"

    # 영업이익률 평가 (높을수록 좋음, 산업별로 다름)
    if latest_data["영업이익률"] >= 15:
        financial_health["영업이익률 상태"] = "매우 좋음"
    elif latest_data["영업이익률"] >= 10:
        financial_health["영업이익률 상태"] = "좋음"
    elif latest_data["영업이익률"] >= 5:
        financial_health["영업이익률 상태"] = "보통"
    else:
        financial_health["영업이익률 상태"] = "주의"

    # 종합 평가
    status_map = {"매우 좋음": 4, "좋음": 3, "보통": 2, "주의": 1, "위험": 0}

    total_score = sum(status_map.get(status, 0) for status in financial_health.values())
    avg_score = total_score / len(financial_health)

    if avg_score >= 3.5:
        financial_health["종합 평가"] = "매우 좋음"
    elif avg_score >= 2.5:
        financial_health["종합 평가"] = "좋음"
    elif avg_score >= 1.5:
        financial_health["종합 평가"] = "보통"
    elif avg_score >= 0.5:
        financial_health["종합 평가"] = "주의"
    else:
        financial_health["종합 평가"] = "위험"

    print("\n===== 재무제표 분석 결과 =====\n")
    print("[손익계산서]")
    print(income_statement)
    print("\n[대차대조표]")
    print(balance_sheet)
    print("\n[현금흐름표]")
    print(cash_flow)
    print("\n[성장률 분석]")
    print(growth_rates)
    print("\n[재무건전성 평가]")
    print(
        f"부채비율: {latest_data['부채비율']:.2f}% ({financial_health['부채비율 상태']})"
    )
    print(
        f"유동비율: {latest_data['유동비율']:.2f}% ({financial_health['유동비율 상태']})"
    )
    print(
        f"영업이익률: {latest_data['영업이익률']:.2f}% ({financial_health['영업이익률 상태']})"
    )
    print(f"\n종합 평가: {financial_health['종합 평가']}")

    return {
        "income_statement": income_statement,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
        "growth_rates": growth_rates,
        "financial_health": financial_health,
        "company_name": company_name,  # 회사명 추가
    }


def analyze_by_stock_code(api_key, stock_code):
    """
    종목 코드로 기업을 찾아 재무제표를 분석합니다.
    """
    # 상장기업 목록 가져오기
    print("상장기업 목록을 가져오는 중...")
    companies_df = fetch_corp_codes(api_key)
    if companies_df.empty:
        print("상장기업 목록을 가져오는데 실패했습니다.")
        return

    # 종목 코드로 기업 코드 찾기
    corp_code = find_corp_code_by_stock_code(companies_df, stock_code)
    if not corp_code:
        return

    # 기업 정보 출력
    company_info = companies_df[companies_df["corp_code"] == corp_code].iloc[0]
    print(
        f"\n선택한 기업: {company_info['corp_name']} " f"({company_info['stock_code']})"
    )
    print(f"기업 코드: {corp_code}")

    # 최근 5개년 재무제표 데이터 가져오기
    current_year = datetime.today().year
    years = range(current_year - 4, current_year + 1)

    financial_data_by_year = {}
    for year in years:
        print(f"\n{year}년 재무제표 데이터를 가져오는 중...")
        financial_data = fetch_financial_statements(api_key, corp_code, year)
        if financial_data:
            print(f"{year}년 데이터 항목 수: {len(financial_data)}")
            # 데이터 샘플 출력 (처음 3개 항목)
            for i, item in enumerate(financial_data[:3]):
                print(
                    f"  항목 {i+1}: {item.get('account_nm')} = "
                    f"{item.get('thstrm_amount')}"
                )

        financial_analysis = analyze_financial_statements(stock_code)
        if financial_analysis:
            financial_data_by_year[year] = financial_analysis
        else:
            print(f"{year}년 데이터 분석 실패")

    if not financial_data_by_year:
        print("재무제표 데이터를 가져오는데 실패했습니다.")
        return

    # 재무제표 데이터 출력
    print("\n===== 재무제표 분석 결과 =====")

    # 데이터프레임으로 변환
    df = pd.DataFrame(financial_data_by_year).T

    # 주요 지표 선택
    key_indicators = [
        "매출액",
        "영업이익",
        "당기순이익",
        "영업이익률",
        "순이익률",
        "자산총계",
        "부채총계",
        "자본총계",
        "부채비율",
        "유동비율",
        "영업활동현금흐름",
        "투자활동현금흐름",
        "재무활동현금흐름",
    ]

    # 존재하는 지표만 선택
    available_indicators = [col for col in key_indicators if col in df.columns]

    if not available_indicators:
        print("분석 가능한 재무지표가 없습니다.")
        return

    # 선택된 지표만 출력
    selected_df = df[available_indicators]

    # 데이터 출력
    if "매출액" in available_indicators:
        print("\n[손익계산서]")
        income_statement_cols = [
            col
            for col in ["매출액", "영업이익", "당기순이익", "영업이익률", "순이익률"]
            if col in available_indicators
        ]
        print(selected_df[income_statement_cols])

    if "자산총계" in available_indicators:
        print("\n[대차대조표]")
        balance_sheet_cols = [
            col
            for col in ["자산총계", "부채총계", "자본총계", "부채비율", "유동비율"]
            if col in available_indicators
        ]
        print(selected_df[balance_sheet_cols])

    if "영업활동현금흐름" in available_indicators:
        print("\n[현금흐름표]")
        cash_flow_cols = [
            col
            for col in ["영업활동현금흐름", "투자활동현금흐름", "재무활동현금흐름"]
            if col in available_indicators
        ]
        print(selected_df[cash_flow_cols])

    # 성장률 계산
    print("\n[성장률 분석]")
    growth_rates = {}
    for year in years[1:]:
        prev_year = year - 1
        if prev_year in financial_data_by_year and year in financial_data_by_year:
            # 0으로 나누기 방지
            prev_revenue = financial_data_by_year[prev_year]["매출액"]
            prev_operating_income = financial_data_by_year[prev_year]["영업이익"]
            prev_net_income = financial_data_by_year[prev_year]["당기순이익"]

            curr_revenue = financial_data_by_year[year]["매출액"]
            curr_operating_income = financial_data_by_year[year]["영업이익"]
            curr_net_income = financial_data_by_year[year]["당기순이익"]

            # 0으로 나누기 방지
            revenue_growth = 0
            if prev_revenue != 0:
                revenue_growth = ((curr_revenue / prev_revenue) - 1) * 100

            operating_income_growth = 0
            if prev_operating_income != 0:
                operating_income_growth = (
                    (curr_operating_income / prev_operating_income) - 1
                ) * 100

            net_income_growth = 0
            if prev_net_income != 0:
                net_income_growth = ((curr_net_income / prev_net_income) - 1) * 100

            growth_rates[year] = {
                "매출액 성장률": revenue_growth,
                "영업이익 성장률": operating_income_growth,
                "순이익 성장률": net_income_growth,
            }

    if growth_rates:
        growth_df = pd.DataFrame(growth_rates).T
        print(growth_df)

    # 재무건전성 평가
    print("\n[재무건전성 평가]")
    latest_year = max(financial_data_by_year.keys())
    latest_data = financial_data_by_year[latest_year]

    # 부채비율 평가
    debt_ratio = latest_data["부채비율"]
    debt_ratio_evaluation = (
        "양호" if debt_ratio < 200 else "주의" if debt_ratio < 400 else "위험"
    )

    # 유동비율 평가
    current_ratio = latest_data["유동비율"]
    current_ratio_evaluation = (
        "양호" if current_ratio > 200 else "주의" if current_ratio > 150 else "위험"
    )

    # 영업이익률 평가
    operating_margin = latest_data["영업이익률"]
    operating_margin_evaluation = (
        "양호" if operating_margin > 10 else "주의" if operating_margin > 5 else "위험"
    )

    print(f"부채비율: {debt_ratio:.2f}% ({debt_ratio_evaluation})")
    print(f"유동비율: {current_ratio:.2f}% ({current_ratio_evaluation})")
    print(f"영업이익률: {operating_margin:.2f}% ({operating_margin_evaluation})")

    # 종합 평가
    risk_count = sum(
        1
        for eval in [
            debt_ratio_evaluation,
            current_ratio_evaluation,
            operating_margin_evaluation,
        ]
        if eval == "위험"
    )
    caution_count = sum(
        1
        for eval in [
            debt_ratio_evaluation,
            current_ratio_evaluation,
            operating_margin_evaluation,
        ]
        if eval == "주의"
    )

    if risk_count >= 2:
        overall_evaluation = "위험"
    elif risk_count == 1 or caution_count >= 2:
        overall_evaluation = "주의"
    else:
        overall_evaluation = "양호"

    print(f"\n종합 평가: {overall_evaluation}")


def main():
    # API 키 확인 또는 등록
    api_key = get_api_key()
    if not api_key:
        print("DART API 키가 등록되어 있지 않습니다.")
        api_key = input("DART API 키를 입력하세요: ")
        if register_api_key(api_key):
            print("API 키가 성공적으로 등록되었습니다.")
        else:
            print("API 키 등록에 실패했습니다.")
            return

    # 분석 방식 선택
    print("\n분석 방식을 선택하세요:")
    print("1. 기업명으로 분석")
    print("2. 종목 코드로 분석")
    choice = input("선택 (1 또는 2): ").strip()

    if choice == "1":
        # 상장기업 목록 가져오기
        print("상장기업 목록을 가져오는 중...")
        companies_df = fetch_corp_codes(api_key)
        if companies_df.empty:
            print("상장기업 목록을 가져오는데 실패했습니다.")
            return

        # 사용자에게 기업명 입력 요청
        corp_name = input("\n분석할 기업명을 입력하세요: ")

        # 기업명으로 기업 코드 찾기
        corp_code = find_corp_code_by_name(companies_df, corp_name)
        if not corp_code:
            print(f"'{corp_name}' 기업을 찾을 수 없습니다.")
            return

        # 기업 정보 출력
        company_info = companies_df[companies_df["corp_code"] == corp_code].iloc[0]
        print(
            f"\n선택한 기업: {company_info['corp_name']} "
            f"({company_info['stock_code']})"
        )
        print(f"기업 코드: {corp_code}")

        # 최근 5개년 재무제표 데이터 가져오기
        current_year = datetime.today().year
        years = range(current_year - 4, current_year + 1)

        financial_data_by_year = {}
        for year in years:
            print(f"\n{year}년 재무제표 데이터를 가져오는 중...")
            financial_data = fetch_financial_statements(api_key, corp_code, year)
            if financial_data:
                print(f"{year}년 데이터 항목 수: {len(financial_data)}")
                # 데이터 샘플 출력 (처음 3개 항목)
                for i, item in enumerate(financial_data[:3]):
                    print(
                        f"  항목 {i+1}: {item.get('account_nm')} = "
                        f"{item.get('thstrm_amount')}"
                    )

            financial_analysis = analyze_financial_statements(
                company_info["stock_code"]
            )
            if financial_analysis:
                financial_data_by_year[year] = financial_analysis
            else:
                print(f"{year}년 데이터 분석 실패")

        if not financial_data_by_year:
            print("재무제표 데이터를 가져오는데 실패했습니다.")
            return

        # 재무제표 데이터 출력
        print("\n===== 재무제표 분석 결과 =====")

        # 데이터프레임으로 변환
        df = pd.DataFrame(financial_data_by_year).T

        # 주요 지표 선택
        key_indicators = [
            "매출액",
            "영업이익",
            "당기순이익",
            "영업이익률",
            "순이익률",
            "자산총계",
            "부채총계",
            "자본총계",
            "부채비율",
            "유동비율",
            "영업활동현금흐름",
            "투자활동현금흐름",
            "재무활동현금흐름",
        ]

        # 존재하는 지표만 선택
        available_indicators = [col for col in key_indicators if col in df.columns]

        if not available_indicators:
            print("분석 가능한 재무지표가 없습니다.")
            return

        # 선택된 지표만 출력
        selected_df = df[available_indicators]

        # 데이터 출력
        if "매출액" in available_indicators:
            print("\n[손익계산서]")
            income_statement_cols = [
                col
                for col in [
                    "매출액",
                    "영업이익",
                    "당기순이익",
                    "영업이익률",
                    "순이익률",
                ]
                if col in available_indicators
            ]
            print(selected_df[income_statement_cols])

        if "자산총계" in available_indicators:
            print("\n[대차대조표]")
            balance_sheet_cols = [
                col
                for col in ["자산총계", "부채총계", "자본총계", "부채비율", "유동비율"]
                if col in available_indicators
            ]
            print(selected_df[balance_sheet_cols])

        if "영업활동현금흐름" in available_indicators:
            print("\n[현금흐름표]")
            cash_flow_cols = [
                col
                for col in ["영업활동현금흐름", "투자활동현금흐름", "재무활동현금흐름"]
                if col in available_indicators
            ]
            print(selected_df[cash_flow_cols])

        # 성장률 계산
        print("\n[성장률 분석]")
        growth_rates = {}
        for year in years[1:]:
            prev_year = year - 1
            if prev_year in financial_data_by_year and year in financial_data_by_year:
                # 0으로 나누기 방지
                prev_revenue = financial_data_by_year[prev_year]["매출액"]
                prev_operating_income = financial_data_by_year[prev_year]["영업이익"]
                prev_net_income = financial_data_by_year[prev_year]["당기순이익"]

                curr_revenue = financial_data_by_year[year]["매출액"]
                curr_operating_income = financial_data_by_year[year]["영업이익"]
                curr_net_income = financial_data_by_year[year]["당기순이익"]

                # 0으로 나누기 방지
                revenue_growth = 0
                if prev_revenue != 0:
                    revenue_growth = ((curr_revenue / prev_revenue) - 1) * 100

                operating_income_growth = 0
                if prev_operating_income != 0:
                    operating_income_growth = (
                        (curr_operating_income / prev_operating_income) - 1
                    ) * 100

                net_income_growth = 0
                if prev_net_income != 0:
                    net_income_growth = ((curr_net_income / prev_net_income) - 1) * 100

                growth_rates[year] = {
                    "매출액 성장률": revenue_growth,
                    "영업이익 성장률": operating_income_growth,
                    "순이익 성장률": net_income_growth,
                }

        if growth_rates:
            growth_df = pd.DataFrame(growth_rates).T
            print(growth_df)

        # 재무건전성 평가
        print("\n[재무건전성 평가]")
        latest_year = max(financial_data_by_year.keys())
        latest_data = financial_data_by_year[latest_year]

        # 부채비율 평가
        debt_ratio = latest_data["부채비율"]
        debt_ratio_evaluation = (
            "양호" if debt_ratio < 200 else "주의" if debt_ratio < 400 else "위험"
        )

        # 유동비율 평가
        current_ratio = latest_data["유동비율"]
        current_ratio_evaluation = (
            "양호" if current_ratio > 200 else "주의" if current_ratio > 150 else "위험"
        )

        # 영업이익률 평가
        operating_margin = latest_data["영업이익률"]
        operating_margin_evaluation = (
            "양호"
            if operating_margin > 10
            else "주의" if operating_margin > 5 else "위험"
        )

        print(f"부채비율: {debt_ratio:.2f}% ({debt_ratio_evaluation})")
        print(f"유동비율: {current_ratio:.2f}% ({current_ratio_evaluation})")
        print(f"영업이익률: {operating_margin:.2f}% ({operating_margin_evaluation})")

        # 종합 평가
        risk_count = sum(
            1
            for eval in [
                debt_ratio_evaluation,
                current_ratio_evaluation,
                operating_margin_evaluation,
            ]
            if eval == "위험"
        )
        caution_count = sum(
            1
            for eval in [
                debt_ratio_evaluation,
                current_ratio_evaluation,
                operating_margin_evaluation,
            ]
            if eval == "주의"
        )

        if risk_count >= 2:
            overall_evaluation = "위험"
        elif risk_count == 1 or caution_count >= 2:
            overall_evaluation = "주의"
        else:
            overall_evaluation = "양호"

        print(f"\n종합 평가: {overall_evaluation}")

    elif choice == "2":
        # 사용자에게 종목 코드 입력 요청
        stock_code = input("\n분석할 종목 코드를 입력하세요 (예: 005930): ")

        # 종목 코드로 분석
        analyze_by_stock_code(api_key, stock_code)

    else:
        print("잘못된 선택입니다. 1 또는 2를 입력하세요.")


if __name__ == "__main__":
    main()
