# 주식 종합 분석 도구(OpenAI API)

"""
주식 종합 분석 도구
재무 분석과 기술적 분석을 결합하고 OpenAI API를 활용하여 종합적인 주식 분석을 제공합니다.
"""

import json
import os
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import openai
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from matplotlib import font_manager, gridspec

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print(
        "경고: OPENAI_API_KEY가 설정되지 않았습니다. AI 분석 기능이 작동하지 않을 수 있습니다."
    )
else:
    openai.api_key = openai_api_key

# 한글 폰트 설정
mac_font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
if os.path.exists(mac_font_path):
    font_manager.fontManager.addfont(mac_font_path)
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
    plt.rcParams["font.sans-serif"] = ["AppleGothic"]
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.unicode_minus"] = False
    print(f"한글 폰트 설정 완료: {mac_font_path}")
else:
    print("한글 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")

# 로케일 설정
plt.rcParams["axes.formatter.use_locale"] = True


def fetch_stock_data(stock_code, start_date=None, end_date=None):
    """
    Yahoo Finance API를 사용하여 주식 데이터를 가져옵니다.
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Yahoo Finance에서 {stock_code}.KS 데이터를 가져오는 중...")

    # yfinance Ticker 객체 생성
    ticker = yf.Ticker(f"{stock_code}.KS")
    df = ticker.history(start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"데이터를 가져올 수 없습니다: {stock_code}")

    print(f"데이터 가져오기 완료: {len(df)}개의 데이터")

    # 컬럼 이름 변경
    df = df.rename(
        columns={
            "Adj Close": "adj_close",
            "Close": "close",
            "High": "high",
            "Low": "low",
            "Open": "open",
            "Volume": "volume",
        }
    )

    # 인덱스 처리
    df = df.reset_index()
    df = df.rename(columns={"Date": "date"})

    return df


def fetch_financial_data(stock_code):
    """
    Yahoo Finance API를 사용하여 재무 데이터를 가져옵니다.
    """
    print(f"Yahoo Finance에서 {stock_code}.KS 재무 데이터를 가져오는 중...")

    # yfinance Ticker 객체 생성
    ticker = yf.Ticker(f"{stock_code}.KS")

    # 재무제표 데이터 가져오기
    try:
        income_stmt = ticker.income_stmt
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow

        print("재무 데이터 가져오기 완료")

        return {
            "income_stmt": income_stmt,
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow,
        }
    except Exception as e:
        print(f"재무 데이터를 가져오는 중 오류 발생: {e}")
        return None


def calculate_bollinger_bands(df, window=20, num_std=2):
    """
    볼린저 밴드를 계산합니다.
    """
    df = df.copy()

    # 볼린저 밴드 계산
    middle = df["close"].rolling(window=window).mean()
    std_dev = df["close"].rolling(window=window).std()

    # 볼린저 밴드 값 계산
    df["bb_middle"] = middle
    df["bb_upper"] = middle + (std_dev * num_std)
    df["bb_lower"] = middle - (std_dev * num_std)

    # 상태 계산
    df["bb_status"] = "중립"
    df.loc[df["close"] > df["bb_upper"], "bb_status"] = "과매수"
    df.loc[df["close"] < df["bb_lower"], "bb_status"] = "과매도"

    return df


def calculate_technical_indicators(df):
    """
    기술적 지표를 계산합니다.
    """
    df = df.copy()

    # 이동평균 계산
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()

    # MA 상태 계산
    df["ma_status"] = "중립"
    df.loc[df["ma5"] > df["ma20"], "ma_status"] = "골든크로스"
    df.loc[df["ma5"] < df["ma20"], "ma_status"] = "데드크로스"

    # RSI 계산
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # RSI 상태 계산
    df["rsi_status"] = "중립"
    df.loc[df["rsi"] > 70, "rsi_status"] = "과매수"
    df.loc[df["rsi"] < 30, "rsi_status"] = "과매도"

    # MACD 계산
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["signal"]

    # MACD 상태 계산
    df["macd_status"] = "중립"
    df.loc[df["macd"] > df["signal"], "macd_status"] = "상승"
    df.loc[df["macd"] < df["signal"], "macd_status"] = "하락"

    # 볼린저 밴드 계산
    df = calculate_bollinger_bands(df)

    return df


def analyze_financials(financial_data):
    """
    재무 데이터를 분석합니다.
    """
    if not financial_data:
        return "재무 데이터를 분석할 수 없습니다."

    income_stmt = financial_data["income_stmt"]
    balance_sheet = financial_data["balance_sheet"]
    cash_flow = financial_data["cash_flow"]

    analysis = []

    # 손익계산서 분석
    if not income_stmt.empty:
        try:
            # 최근 4분기 데이터
            recent_income = income_stmt.iloc[:, 0:4]

            # 매출액 추이
            if "Total Revenue" in recent_income.index:
                revenue = recent_income.loc["Total Revenue"]
                revenue_growth = revenue.pct_change().mean() * 100
                analysis.append(f"매출액 성장률: {revenue_growth:.2f}%")

            # 영업이익 추이
            if "Operating Income" in recent_income.index:
                op_income = recent_income.loc["Operating Income"]
                op_income_growth = op_income.pct_change().mean() * 100
                analysis.append(f"영업이익 성장률: {op_income_growth:.2f}%")

            # 순이익 추이
            if "Net Income" in recent_income.index:
                net_income = recent_income.loc["Net Income"]
                net_income_growth = net_income.pct_change().mean() * 100
                analysis.append(f"순이익 성장률: {net_income_growth:.2f}%")
        except Exception as e:
            analysis.append(f"손익계산서 분석 중 오류: {e}")

    # 대차대조표 분석
    if not balance_sheet.empty:
        try:
            # 최근 4분기 데이터
            recent_balance = balance_sheet.iloc[:, 0:4]

            # 유동자산과 유동부채
            if (
                "Total Current Assets" in recent_balance.index
                and "Total Current Liabilities" in recent_balance.index
            ):
                current_assets = recent_balance.loc["Total Current Assets"].iloc[0]
                current_liabilities = recent_balance.loc[
                    "Total Current Liabilities"
                ].iloc[0]
                current_ratio = current_assets / current_liabilities
                analysis.append(f"유동비율: {current_ratio:.2f}")

            # 부채비율
            if (
                "Total Liab" in recent_balance.index
                and "Total Stockholder Equity" in recent_balance.index
            ):
                total_liab = recent_balance.loc["Total Liab"].iloc[0]
                equity = recent_balance.loc["Total Stockholder Equity"].iloc[0]
                debt_ratio = (total_liab / equity) * 100
                analysis.append(f"부채비율: {debt_ratio:.2f}%")
        except Exception as e:
            analysis.append(f"대차대조표 분석 중 오류: {e}")

    # 현금흐름표 분석
    if not cash_flow.empty:
        try:
            # 최근 4분기 데이터
            recent_cash = cash_flow.iloc[:, 0:4]

            # 영업활동 현금흐름
            if "Operating Cash Flow" in recent_cash.index:
                op_cash = recent_cash.loc["Operating Cash Flow"].iloc[0]
                analysis.append(f"영업활동 현금흐름: {op_cash:,.0f}")

            # 투자활동 현금흐름
            if "Investing Cash Flow" in recent_cash.index:
                inv_cash = recent_cash.loc["Investing Cash Flow"].iloc[0]
                analysis.append(f"투자활동 현금흐름: {inv_cash:,.0f}")

            # 재무활동 현금흐름
            if "Financing Cash Flow" in recent_cash.index:
                fin_cash = recent_cash.loc["Financing Cash Flow"].iloc[0]
                analysis.append(f"재무활동 현금흐름: {fin_cash:,.0f}")
        except Exception as e:
            analysis.append(f"현금흐름표 분석 중 오류: {e}")

    return "\n".join(analysis)


def plot_stock_analysis(df, stock_code):
    """
    주식 분석 결과를 시각화합니다.
    """
    plt.style.use("default")
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])

    # 가격 차트와 볼린저 밴드
    ax1 = plt.subplot(gs[0])
    lines1 = []
    lines1.append(ax1.plot(df["date"], df["close"], label="종가", color="black")[0])
    lines1.append(
        ax1.plot(df["date"], df["bb_upper"], label="상단 밴드", color="red", alpha=0.7)[
            0
        ]
    )
    lines1.append(
        ax1.plot(
            df["date"], df["bb_middle"], label="중간 밴드", color="blue", alpha=0.7
        )[0]
    )
    lines1.append(
        ax1.plot(
            df["date"], df["bb_lower"], label="하단 밴드", color="green", alpha=0.7
        )[0]
    )
    lines1.append(
        ax1.plot(
            df["date"], df["ma5"], label="5일 이동평균", color="orange", alpha=0.7
        )[0]
    )
    lines1.append(
        ax1.plot(
            df["date"], df["ma20"], label="20일 이동평균", color="purple", alpha=0.7
        )[0]
    )
    ax1.set_title(f"{stock_code} 주가 분석", fontsize=15, fontfamily="AppleGothic")
    ax1.set_xlabel("날짜", fontsize=10, fontfamily="AppleGothic")
    ax1.set_ylabel("가격 (원)", fontsize=10, fontfamily="AppleGothic")
    ax1.legend(fontsize=8, loc="best", prop={"family": "AppleGothic"})
    ax1.grid(True)

    # Y축 포맷 설정
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # RSI 차트
    ax2 = plt.subplot(gs[1])
    lines2 = []
    lines2.append(ax2.plot(df["date"], df["rsi"], label="RSI", color="blue")[0])
    ax2.axhline(y=70, color="red", linestyle="--", alpha=0.7)
    ax2.axhline(y=30, color="green", linestyle="--", alpha=0.7)
    ax2.set_xlabel("날짜", fontsize=10, fontfamily="AppleGothic")
    ax2.set_ylabel("RSI", fontsize=10, fontfamily="AppleGothic")
    ax2.legend(fontsize=8, loc="best", prop={"family": "AppleGothic"})
    ax2.grid(True)

    # MACD 차트
    ax3 = plt.subplot(gs[2])
    lines3 = []
    lines3.append(ax3.plot(df["date"], df["macd"], label="MACD", color="blue")[0])
    lines3.append(ax3.plot(df["date"], df["signal"], label="시그널", color="red")[0])
    bars = ax3.bar(
        df["date"], df["macd_hist"], label="히스토그램", color="gray", alpha=0.5
    )
    ax3.set_xlabel("날짜", fontsize=10, fontfamily="AppleGothic")
    ax3.set_ylabel("MACD", fontsize=10, fontfamily="AppleGothic")
    ax3.legend(fontsize=8, loc="best", prop={"family": "AppleGothic"})
    ax3.grid(True)

    # Y축 포맷 설정
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # 호버 기능 추가
    def price_format(sel):
        ind = sel.target.index
        date = df["date"].iloc[ind]
        value = sel.target.get_ydata()[ind]
        label = sel.artist.get_label()
        return f'날짜: {date.strftime("%Y-%m-%d")}\n{label}: {value:,.0f}'

    cursor1 = mplcursors.cursor(lines1, hover=True)
    cursor1.connect("add", lambda sel: sel.annotation.set_text(price_format(sel)))

    cursor2 = mplcursors.cursor(lines2, hover=True)
    cursor2.connect("add", lambda sel: sel.annotation.set_text(price_format(sel)))

    cursor3 = mplcursors.cursor(lines3, hover=True)
    cursor3.connect("add", lambda sel: sel.annotation.set_text(price_format(sel)))

    cursor_bars = mplcursors.cursor(bars, hover=True)
    cursor_bars.connect("add", lambda sel: sel.annotation.set_text(price_format(sel)))

    plt.tight_layout()
    plt.savefig(f"result/stock_analysis_{stock_code}.png")
    plt.show()


def get_ai_analysis(stock_code, technical_data, financial_analysis):
    """
    OpenAI API를 사용하여 종합적인 주식 분석을 제공합니다.
    """
    if not openai_api_key:
        return "OpenAI API 키가 설정되지 않아 AI 분석을 수행할 수 없습니다."

    try:
        # 최근 데이터 분석
        latest = technical_data.iloc[-1]
        prev = technical_data.iloc[-2]
        change = latest["close"] - prev["close"]
        change_pct = (change / prev["close"]) * 100

        # 매수/매도 신호 분석
        buy_signals = 0
        sell_signals = 0

        # RSI 기반 신호
        if latest["rsi_status"] == "과매도":
            buy_signals += 1
        elif latest["rsi_status"] == "과매수":
            sell_signals += 1

        # MACD 기반 신호
        if latest["macd_status"] == "상승":
            buy_signals += 1
        elif latest["macd_status"] == "하락":
            sell_signals += 1

        # 볼린저 밴드 기반 신호
        if latest["bb_status"] == "과매도":
            buy_signals += 1
        elif latest["bb_status"] == "과매수":
            sell_signals += 1

        # 종합 분석 결과
        if buy_signals > sell_signals:
            overall_signal = "매수 추천"
        elif sell_signals > buy_signals:
            overall_signal = "매도 추천"
        else:
            overall_signal = "관망 추천"

        # OpenAI API에 전송할 프롬프트 구성
        prompt = f"""
        다음은 {stock_code} 종목에 대한 주식 분석 데이터입니다:

        기술적 분석:
        - 최근 종가: {latest['close']:,.0f}원
        - 전일 대비: {change:,.0f}원 ({change_pct:.2f}%)
        - 5일 이동평균: {latest['ma5']:,.0f}원
        - 20일 이동평균: {latest['ma20']:,.0f}원
        - 이동평균 상태: {latest['ma_status']}
        - RSI: {latest['rsi']:.2f}
        - RSI 상태: {latest['rsi_status']}
        - MACD: {latest['macd']:,.0f}
        - 시그널: {latest['signal']:,.0f}
        - MACD 히스토그램: {latest['macd_hist']:,.0f}
        - MACD 상태: {latest['macd_status']}
        - 볼린저 상단: {latest['bb_upper']:,.0f}원
        - 볼린저 하단: {latest['bb_lower']:,.0f}원
        - 볼린저 밴드 상태: {latest['bb_status']}
        - 매수 신호: {buy_signals}개
        - 매도 신호: {sell_signals}개
        - 종합 분석: {overall_signal}

        재무 분석:
        {financial_analysis}

        위 데이터를 바탕으로 {stock_code} 종목에 대한 종합적인 투자 분석을 제공해주세요.
        다음 항목을 포함해주세요:
        1. 현재 주가의 기술적 분석 평가
        2. 재무 상태 평가
        3. 단기 투자 전략 (1-3개월)
        4. 중기 투자 전략 (3-12개월)
        5. 투자 위험 요소
        6. 최종 투자 의견
        """

        # OpenAI API 호출
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
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
        return f"AI 분석 중 오류 발생: {e}"


def analyze_stock(stock_code):
    """
    주식을 분석하고 결과를 출력합니다.
    """
    # 주가 데이터 가져오기
    df = fetch_stock_data(stock_code)
    df = calculate_technical_indicators(df)

    # 재무 데이터 가져오기
    financial_data = fetch_financial_data(stock_code)
    financial_analysis = analyze_financials(financial_data)

    # 최근 데이터 분석
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    change = latest["close"] - prev["close"]
    change_pct = (change / prev["close"]) * 100

    print("\n=== 기술적 분석 결과 ===")
    print(f"최근 종가: {latest['close']:,.0f}원")
    print(f"전일 대비: {change:,.0f}원 ({change_pct:.2f}%)")
    print(f"5일 이동평균: {latest['ma5']:,.0f}원")
    print(f"20일 이동평균: {latest['ma20']:,.0f}원")
    print(f"이동평균 상태: {latest['ma_status']}")
    print(f"RSI: {latest['rsi']:.2f}")
    print(f"RSI 상태: {latest['rsi_status']}")
    print(f"MACD: {latest['macd']:,.0f}")
    print(f"시그널: {latest['signal']:,.0f}")
    print(f"MACD 히스토그램: {latest['macd_hist']:,.0f}")
    print(f"MACD 상태: {latest['macd_status']}")
    print(f"볼린저 상단: {latest['bb_upper']:,.0f}원")
    print(f"볼린저 하단: {latest['bb_lower']:,.0f}원")
    print(f"볼린저 밴드 상태: {latest['bb_status']}")

    # 매수/매도 신호 분석
    buy_signals = 0
    sell_signals = 0

    # RSI 기반 신호
    if latest["rsi_status"] == "과매도":
        buy_signals += 1
    elif latest["rsi_status"] == "과매수":
        sell_signals += 1

    # MACD 기반 신호
    if latest["macd_status"] == "상승":
        buy_signals += 1
    elif latest["macd_status"] == "하락":
        sell_signals += 1

    # 볼린저 밴드 기반 신호
    if latest["bb_status"] == "과매도":
        buy_signals += 1
    elif latest["bb_status"] == "과매수":
        sell_signals += 1

    print("\n=== 매매 신호 ===")
    print(f"매수 신호: {buy_signals}개")
    print(f"매도 신호: {sell_signals}개")

    if buy_signals > sell_signals:
        print("종합 분석: 매수 추천")
    elif sell_signals > buy_signals:
        print("종합 분석: 매도 추천")
    else:
        print("종합 분석: 관망 추천")

    print("\n=== 재무 분석 결과 ===")
    print(financial_analysis)

    # AI 분석 결과
    print("\n=== AI 종합 분석 ===")
    ai_analysis = get_ai_analysis(stock_code, df, financial_analysis)
    print(ai_analysis)

    # 차트 그리기
    plot_stock_analysis(df, stock_code)


def main():
    # 분석할 종목 코드 입력
    stock_code = input("분석할 종목 코드를 입력하세요 (예: 005930): ")

    # 주식 종합 분석 수행
    analyze_stock(stock_code)


if __name__ == "__main__":
    main()
