#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
주식 통합 분석 도구
재무 분석과 기술적 분석을 결합하고 OpenAI API를 활용하여 종합적인 주식 분석을 제공합니다.
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import openai
from dotenv import load_dotenv
from financial_analysis import analyze_financial_statements, fetch_corp_codes
from stock_technical_analysis import analyze_technical

# 환경 변수 로드
load_dotenv()

# API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
dart_api_key = os.getenv("DART_API_KEY")

if not openai_api_key:
    print(
        "경고: OPENAI_API_KEY가 설정되지 않았습니다. AI 분석 기능이 작동하지 않을 수 있습니다."
    )
else:
    openai.api_key = openai_api_key

if not dart_api_key:
    print(
        "경고: DART_API_KEY가 설정되지 않았습니다. 재무제표 분석 기능이 작동하지 않을 수 있습니다."
    )


def get_ai_analysis(stock_code, technical_result, financial_result):
    """
    OpenAI API를 사용하여 종합적인 주식 분석을 제공합니다.
    """
    if not openai_api_key:
        return "OpenAI API 키가 설정되지 않아 AI 분석을 수행할 수 없습니다."

    try:
        # OpenAI API에 전송할 프롬프트 구성
        prompt = f"""
        다음은 {stock_code} 종목에 대한 주식 분석 데이터입니다:

        기술적 분석:
        {technical_result}

        재무 분석:
        {financial_result}

        위 데이터를 바탕으로 {stock_code} 종목에 대한 종합적인 투자 분석을 제공해주세요.
        다음 항목을 포함해주세요:
        1. 현재 주가의 기술적 분석 평가
        2. 재무 상태 평가
        3. 단기 투자 전략 (1-3개월)
        4. 중기 투자 전략 (3-12개월)
        5. 투자 위험 요소
        6. 최종 투자 의견
        """

        # OpenAI API 호출 (최신 버전 사용)
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 전문적인 주식 분석가입니다. 주어진 데이터를 바탕으로 객관적이고 전문적인 투자 분석을 제공합니다.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"AI 분석 중 오류 발생: {str(e)}"


def analyze_stock(stock_code):
    """
    주식 종목에 대한 종합 분석을 수행합니다.
    """
    # 기술적 분석 수행
    print("\n기술적 분석 수행 중...")
    technical_result = analyze_technical(stock_code)

    # 재무제표 분석 수행
    print("\n재무제표 분석 수행 중...")
    financial_result = analyze_financial_statements(stock_code)

    if not financial_result or not technical_result:
        print("분석에 실패했습니다.")
        return None

    # AI 종합 분석 결과 출력
    print("\n=== AI 종합 분석 ===")
    print(f"### {stock_code} 종목 종합적인 투자 분석\n")

    # 1. 기술적 분석 평가
    print("1. **기술적 분석 평가:**")
    print(f"    - 최근 주가: {technical_result['current_price']:,}원")
    print(
        f"    - 가격 변동: {technical_result['price_change']:,}원 "
        f"({technical_result['price_change_percent']:.1f}%)"
    )
    print(f"    - 이동평균 상태: {technical_result['ma_status']}")
    print(
        f"    - RSI: {technical_result['rsi']:.2f} "
        f"({technical_result['rsi_status']})"
    )
    print(f"    - MACD 지표: {technical_result['macd_status']}")
    print(f"    - 볼린저 밴드 상태: {technical_result['bollinger_status']}")
    print(
        f"    - 매수 신호: {technical_result['buy_signals']}개, "
        f"매도 신호: {technical_result['sell_signals']}개"
    )
    print(f"    - 종합 평가: {technical_result['recommendation']}\n")

    # 2. 재무 상태 평가
    print("2. **재무 상태 평가:**")
    latest_year = financial_result["income_statement"].index[-1]
    income_statement = financial_result["income_statement"].loc[latest_year]
    balance_sheet = financial_result["balance_sheet"].loc[latest_year]
    growth_rates = financial_result["growth_rates"].loc[latest_year]

    print(
        f"    - 매출액: {income_statement['매출액']:,.0f}원 "
        f"(전년 대비 {growth_rates['매출액 성장률']:.2f}% 성장)"
    )
    print(
        f"    - 영업이익: {income_statement['영업이익']:,.0f}원 "
        f"(전년 대비 {growth_rates['영업이익 성장률']:.2f}% 성장)"
    )
    print(
        f"    - 당기순이익: {income_statement['당기순이익']:,.0f}원 "
        f"(전년 대비 {growth_rates['당기순이익 성장률']:.2f}% 성장)"
    )
    print(
        f"    - 부채비율: {balance_sheet['부채비율']:.2f}% "
        f"({financial_result['financial_health']['부채비율 상태']})"
    )
    print(
        f"    - 유동비율: {balance_sheet['유동비율']:.2f}% "
        f"({financial_result['financial_health']['유동비율 상태']})"
    )
    print(
        f"    - 영업이익률: {income_statement['영업이익률']:.2f}% "
        f"({financial_result['financial_health']['영업이익률 상태']})"
    )
    print(f"    - 종합 평가: {financial_result['financial_health']['종합 평가']}\n")

    # 3. 단기 투자 전략
    print("3. **단기 투자 전략 (1-3개월):**")
    if technical_result["recommendation"] == "매수 추천":
        print(
            "    - 기술적 분석 결과 매수 신호가 강하며, "
            "단기적인 상승 추세가 예상됩니다."
        )
    elif technical_result["recommendation"] == "매도 추천":
        print(
            "    - 기술적 분석 결과 매도 신호가 강하며, "
            "단기적인 하락 추세가 예상됩니다."
        )
    else:
        print(
            "    - 현재 뚜렷한 매매 신호가 없으며, " "단기적으로는 관망이 권장됩니다."
        )

    if financial_result["financial_health"]["종합 평가"] in ["매우 좋음", "좋음"]:
        print("    - 재무 상태가 양호하여 단기적인 리스크는 제한적입니다.\n")
    elif financial_result["financial_health"]["종합 평가"] == "보통":
        print("    - 재무 상태가 보통이므로 적절한 리스크 관리가 필요합니다.\n")
    else:
        print("    - 재무 상태가 좋지 않아 단기적인 리스크가 높을 수 있습니다.\n")

    # 4. 중기 투자 전략
    print("4. **중기 투자 전략 (3-12개월):**")
    if financial_result["financial_health"]["종합 평가"] in ["매우 좋음", "좋음"]:
        print("    - 재무 건전성이 양호하여 중장기 투자에 적합한 종목입니다.")
        if all(growth_rates > 0):
            print("    - 전년 대비 성장세를 보이고 있어 긍정적인 전망이 가능합니다.\n")
        else:
            print("    - 성장세가 둔화되고 있어 모니터링이 필요합니다.\n")
    else:
        print("    - 재무 건전성이 다소 약해 중장기 투자 시 주의가 필요합니다.\n")

    # 5. 투자 위험 요소
    print("5. **투자 위험 요소:**")
    risks = []
    if technical_result["ma_status"] == "데드크로스":
        risks.append("주가가 하락 추세에 있습니다.")
    if technical_result["rsi"] > 70:
        risks.append("RSI가 과매수 구간에 있어 조정 가능성이 있습니다.")
    if technical_result["rsi"] < 30:
        risks.append("RSI가 과매도 구간에 있습니다.")
    if balance_sheet["부채비율"] > 200:
        risks.append("부채비율이 높아 재무 리스크가 있습니다.")
    if balance_sheet["유동비율"] < 150:
        risks.append("유동비율이 낮아 단기 지급능력에 불안요소가 있습니다.")
    if income_statement["영업이익률"] < 5:
        risks.append("영업이익률이 낮아 수익성에 우려가 있습니다.")

    if risks:
        for risk in risks:
            print(f"    - {risk}")
    else:
        print("    - 현재 특별한 위험 요소가 발견되지 않았습니다.")
    print()

    # 6. 최종 투자 의견
    print("6. **최종 투자 의견:**")
    if technical_result["recommendation"] == "매수 추천" and financial_result[
        "financial_health"
    ]["종합 평가"] in ["매우 좋음", "좋음"]:
        print(
            "    - 기술적 분석과 재무 상태가 모두 양호하여 "
            "투자를 고려해볼 만한 종목입니다."
        )
    elif technical_result["recommendation"] == "매도 추천" or financial_result[
        "financial_health"
    ]["종합 평가"] in ["주의", "위험"]:
        print(
            "    - 현재는 투자를 피하고 상황이 개선될 때까지 "
            "기다리는 것이 좋겠습니다."
        )
    else:
        print("    - 현재는 관망하면서 추가적인 모니터링이 필요한 상황입니다.")
    print()

    print(f"=== {stock_code} 종목 분석 완료 ===\n")
    return {
        "technical_analysis": technical_result,
        "financial_analysis": financial_result,
        "company_name": financial_result.get("company_name", stock_code),
    }


def save_analysis_result(stock_code, result):
    """
    분석 결과를 파일로 저장합니다.
    """
    # 결과 저장
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    current_date = datetime.now().strftime("%Y%m%d")
    company_name = result.get("company_name", stock_code)
    result_file = os.path.join(
        result_dir, f"{company_name}({stock_code})_{current_date}.txt"
    )

    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"=== {stock_code} 종목 분석 결과 ===\n\n")

        # 기술적 분석 결과
        f.write("1. 기술적 분석\n")
        f.write("-" * 50 + "\n")
        f.write(f"현재가: {result['technical_analysis']['current_price']:,}원\n")
        f.write(
            f"전일대비: {result['technical_analysis']['price_change']:+,}원 ({result['technical_analysis']['price_change_percent']:+.2f}%)\n"
        )
        f.write(f"이동평균 상태: {result['technical_analysis']['ma_status']}\n")
        f.write(
            f"RSI: {result['technical_analysis']['rsi']:.2f} ({result['technical_analysis']['rsi_status']})\n"
        )
        f.write(f"MACD 지표: {result['technical_analysis']['macd']:.2f}\n")
        f.write(f"MACD 시그널: {result['technical_analysis']['signal']:.2f}\n")
        f.write(f"MACD 히스토그램: {result['technical_analysis']['macd_hist']:.2f}\n")
        f.write(f"MACD 상태: {result['technical_analysis']['macd_status']}\n")
        f.write(
            f"볼린저 밴드 상태: {result['technical_analysis']['bollinger_status']}\n"
        )
        f.write(f"매매 추천: {result['technical_analysis']['recommendation']}\n\n")

        # 재무제표 분석 결과
        f.write("2. 재무제표 분석\n")
        f.write("-" * 50 + "\n")
        financial = result["financial_analysis"]
        latest_bs = financial["balance_sheet"].iloc[-1]
        latest_is = financial["income_statement"].iloc[-1]

        f.write(f"유동자산: {latest_bs['유동자산']:,}원\n")
        f.write(f"자산총계: {latest_bs['자산총계']:,}원\n")
        f.write(f"유동부채: {latest_bs['유동부채']:,}원\n")
        f.write(f"부채총계: {latest_bs['부채총계']:,}원\n")
        f.write(f"자본총계: {latest_bs['자본총계']:,}원\n")
        f.write(f"매출액: {latest_is['매출액']:,}원\n")
        f.write(f"영업이익: {latest_is['영업이익']:,}원\n")
        f.write(f"당기순이익: {latest_is['당기순이익']:,}원\n\n")

        # 재무건전성 평가
        f.write("3. 재무건전성 평가\n")
        f.write("-" * 50 + "\n")
        f.write(f"유동비율: {latest_bs['유동비율']:.2f}%\n")
        f.write(f"부채비율: {latest_bs['부채비율']:.2f}%\n")
        f.write(f"영업이익률: {latest_is['영업이익률']:.2f}%\n")
        f.write(f"재무건전성 등급: {financial['financial_health']['종합 평가']}\n\n")

        # 종합 평가
        f.write("4. 종합 평가\n")
        f.write("-" * 50 + "\n")
        f.write(f"기술적 분석: {result['technical_analysis']['recommendation']}\n")
        f.write(f"재무건전성: {financial['financial_health']['종합 평가']}\n\n")

        # LLM을 활용한 상세 분석
        analysis_prompt = f"""
        다음은 {company_name}({stock_code}) 종목의 종합 분석 결과입니다:

        1. 가격 정보:
        - 현재가: {result['technical_analysis']['current_price']:,}원
        - 전일대비: {result['technical_analysis']['price_change']:+.2f}%
        - 거래량: {result['technical_analysis'].get('volume', 'N/A')}
        - 시가총액: {result['technical_analysis'].get('market_cap', 'N/A')}원

        2. 수익성 지표:
        - PER: {result['technical_analysis'].get('per', 'N/A')}
        - PBR: {result['technical_analysis'].get('pbr', 'N/A')}
        - ROE: {result['technical_analysis'].get('roe', 'N/A')}%
        - ROA: {result['technical_analysis'].get('roa', 'N/A')}%
        - 영업이익률: {result['technical_analysis'].get('operating_margin', 'N/A')}%
        - 순이익률: {result['technical_analysis'].get('net_profit_margin', 'N/A')}%

        3. 재무건전성:
        - 부채비율: {result['technical_analysis'].get('debt_ratio', 'N/A')}%
        - 유동비율: {result['technical_analysis'].get('current_ratio', 'N/A')}%

        4. 재무제표:
        - 당기순이익: {result['technical_analysis'].get('net_income', 'N/A')}원
        - 영업이익: {result['technical_analysis'].get('operating_income', 'N/A')}원
        - 매출액: {result['technical_analysis'].get('revenue', 'N/A')}원
        - 자산총계: {result['technical_analysis'].get('total_assets', 'N/A')}원
        - 부채총계: {result['technical_analysis'].get('total_liabilities', 'N/A')}원
        - 자본총계: {result['technical_analysis'].get('total_equity', 'N/A')}원

        5. 현금흐름:
        - 영업활동현금흐름: {result['technical_analysis'].get('operating_cash_flow', 'N/A')}원
        - 투자활동현금흐름: {result['technical_analysis'].get('investing_cash_flow', 'N/A')}원
        - 재무활동현금흐름: {result['technical_analysis'].get('financing_cash_flow', 'N/A')}원

        6. 성장성:
        - 매출액성장률: {result['technical_analysis'].get('revenue_growth', 'N/A')}%
        - 영업이익성장률: {result['technical_analysis'].get('operating_income_growth', 'N/A')}%
        - 순이익성장률: {result['technical_analysis'].get('net_income_growth', 'N/A')}%
        - 자산성장률: {result['technical_analysis'].get('total_assets_growth', 'N/A')}%
        - 부채성장률: {result['technical_analysis'].get('total_liabilities_growth', 'N/A')}%
        - 자본성장률: {result['technical_analysis'].get('total_equity_growth', 'N/A')}%
        - 영업활동현금흐름성장률: {result['technical_analysis'].get('operating_cash_flow_growth', 'N/A')}%
        - 투자활동현금흐름성장률: {result['technical_analysis'].get('investing_cash_flow_growth', 'N/A')}%
        - 재무활동현금흐름성장률: {result['technical_analysis'].get('financing_cash_flow_growth', 'N/A')}%

        7. 기술적 분석:
        {result['technical_analysis'].get('technical_analysis', '기술적 분석 정보가 없습니다.')}

        8. 재무건전성 평가:
        {result['financial_analysis']['financial_health']['종합 평가']}

        위 데이터를 바탕으로 다음 항목에 대해 상세히 분석해주세요:

        1. 종목의 전반적인 투자 매력도
        2. 수익성 분석 (PER, ROE, 영업이익률 등)
        3. 재무건전성 분석 (부채비율, 유동비율 등)
        4. 성장성 분석 (매출액, 영업이익, 순이익 성장률 등)
        5. 현금흐름 분석 (영업/투자/재무활동 현금흐름)
        6. 기술적 분석 결과 해석
        7. 투자 위험 요소
        8. 단기/중기/장기 투자 관점에서의 투자 의견
        9. 투자 시 주의해야 할 점
        10. 종합 투자 의견

        각 항목별로 구체적인 수치와 근거를 들어 설명해주시고, 투자 결정에 도움이 되는 실질적인 조언을 제공해주세요.
        """

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 주식 투자 분석 전문가입니다. 주어진 데이터를 바탕으로 종목의 투자 가치를 객관적이고 전문적으로 분석해주세요.",
                    },
                    {"role": "user", "content": analysis_prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )

            detailed_analysis = response.choices[0].message.content
            print("\n===== 상세 분석 결과 =====")
            print(detailed_analysis)

            # 결과 저장
            f.write("5. 상세 분석 결과\n")
            f.write("-" * 50 + "\n")
            f.write(detailed_analysis)

        except Exception as e:
            print(f"상세 분석 중 오류 발생: {e}")

    print(f"\n분석 결과가 {result_file}에 저장되었습니다.")


def main():
    # 분석할 종목 코드 입력
    stock_code = input("분석할 종목 코드를 입력하세요 (예: 005930): ")

    # 주식 종합 분석 수행
    result = analyze_stock(stock_code)

    # 분석 결과 저장
    save_analysis_result(stock_code, result)

    return result


if __name__ == "__main__":
    main()
