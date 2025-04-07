# 주식 기술적 분석(yfinance)

import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import mplcursors
import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf

# 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"  # macOS용 한글 폰트
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# 폰트 설정 확인
print(f"한글 폰트 설정 완료: {plt.rcParams['font.family']}")

# 로케일 설정
plt.rcParams["axes.formatter.use_locale"] = True

# 폰트 설정을 모든 서브플롯에 적용
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.size"] = 10


def fetch_stock_data(stock_code):
    """
    Yahoo Finance에서 주식 데이터를 가져옵니다.
    """
    try:
        print(f"Yahoo Finance에서 {stock_code}.KS 데이터를 가져오는 중...")
        df = yf.download(f"{stock_code}.KS", period="1mo")
        if df is None or df.empty:
            print("데이터를 가져오는데 실패했습니다.")
            return None

        # 컬럼 이름 변경
        df = df.rename(
            columns={
                "Adj Close": "close",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        print(f"데이터 가져오기 완료: {len(df)}개의 데이터")
        return df
    except Exception as e:
        print(f"데이터를 가져오는 중 오류 발생: {e}")
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


def plot_stock_analysis(df, stock_code):
    """주식 분석 결과를 시각화합니다."""
    # 스타일 설정
    mc = mpf.make_marketcolors(
        up="red",
        down="blue",
        edge="inherit",
        wick="inherit",
        volume="in",
        ohlc="inherit",
    )
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle="",
        y_on_right=True,
        figcolor="white",
        facecolor="white",
        edgecolor="black",
        rc={"font.family": "AppleGothic", "axes.unicode_minus": False},
    )

    # 데이터 준비
    df_plot = df.copy()
    df_plot.index = pd.to_datetime(df_plot.index)

    # 이동평균선과 볼린저 밴드 추가
    apdict = [
        mpf.make_addplot(df_plot["ma5"], color="red", width=0.7, label="MA5"),
        mpf.make_addplot(df_plot["ma20"], color="blue", width=0.7, label="MA20"),
        mpf.make_addplot(
            df_plot["bb_upper"], color="green", width=0.7, label="Upper Band"
        ),
        mpf.make_addplot(
            df_plot["bb_lower"], color="green", width=0.7, label="Lower Band"
        ),
    ]

    # MACD 패널 추가
    colors = ["g" if val >= 0 else "r" for val in df_plot["macd_hist"]]
    apdict.extend(
        [
            mpf.make_addplot(
                df_plot["macd"], panel=1, color="blue", width=0.7, label="MACD"
            ),
            mpf.make_addplot(
                df_plot["signal"], panel=1, color="red", width=0.7, label="Signal"
            ),
            mpf.make_addplot(
                df_plot["macd_hist"],
                type="bar",
                width=0.7,
                panel=1,
                color=colors,
                label="Histogram",
            ),
        ]
    )

    # 그래프 그리기
    fig, axes = mpf.plot(
        df_plot,
        type="candle",
        style=s,
        addplot=apdict,
        volume=False,
        panel_ratios=(2, 1),
        figsize=(15, 10),
        returnfig=True,
    )

    # 축 레이블 설정
    axes[0].set_title("주가 분석")
    axes[2].set_title("MACD")

    # y축 가격 포맷 설정
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # Hover 기능 추가
    def price_format(sel):
        """
        그래프의 hover 텍스트를 포맷팅합니다.
        """
        try:
            # 인덱스와 값 가져오기
            idx = sel.target.index
            value = sel.target.get_ydata()[idx]
            date = df_plot.index[idx]
            label = sel.artist.get_label()

            return f'날짜: {date.strftime("%Y-%m-%d")}\n{label}: {value:,.0f}'
        except Exception as e:
            return f"오류: {str(e)}"

    # 각 선에 커서 추가
    for ax in [axes[0], axes[2]]:
        for line in ax.get_lines():
            if line.get_label() not in ["", "_nolegend_"]:
                cursor = mplcursors.cursor(line, hover=True)
                cursor.connect(
                    "add", lambda sel: sel.annotation.set_text(price_format(sel))
                )

    plt.tight_layout()
    return fig


def calculate_rsi(prices, period=14):
    """
    RSI(Relative Strength Index) 계산
    """
    # 가격 변화 계산
    deltas = prices.diff()
    deltas = deltas[1:]  # 첫 번째 NaN 제거

    # 상승/하락 구분
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)

    # 평균 계산
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()

    # RSI 계산
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    MACD(Moving Average Convergence Divergence) 계산
    """
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def analyze_technical(stock_code):
    """
    주식의 기술적 분석을 수행합니다.
    """
    # 데이터 가져오기
    df = fetch_stock_data(stock_code)
    if df is None or len(df) < 20:
        print("충분한 데이터를 가져오지 못했습니다.")
        return None

    # 현재가 및 변동 계산
    current_price = df["close"].iloc[-1].item()
    prev_price = df["close"].iloc[-2].item()
    price_change = current_price - prev_price
    price_change_percent = (price_change / prev_price) * 100

    # 이동평균 계산
    ma5 = df["close"].rolling(window=5).mean()
    ma20 = df["close"].rolling(window=20).mean()
    ma_status = (
        "데드크로스" if ma5.iloc[-1].item() < ma20.iloc[-1].item() else "골든크로스"
    )

    # RSI 계산
    rsi = calculate_rsi(df["close"])
    rsi_value = rsi.iloc[-1].item()
    if rsi_value >= 70:
        rsi_status = "과매수"
    elif rsi_value <= 30:
        rsi_status = "과매도"
    else:
        rsi_status = "중립"

    # MACD 계산
    macd, signal, hist = calculate_macd(df["close"])
    macd_value = macd.iloc[-1].item()
    signal_value = signal.iloc[-1].item()
    hist_value = hist.iloc[-1].item()
    macd_status = "하락" if hist_value < 0 else "상승"

    # 볼린저 밴드 계산
    bb_upper = df["close"].rolling(window=20).mean() + (
        df["close"].rolling(window=20).std() * 2
    )
    bb_lower = df["close"].rolling(window=20).mean() - (
        df["close"].rolling(window=20).std() * 2
    )
    bb_status = "중립"
    if current_price > bb_upper.iloc[-1].item():
        bb_status = "과매수"
    elif current_price < bb_lower.iloc[-1].item():
        bb_status = "과매도"

    # 매매 신호 계산
    buy_signals = 0
    sell_signals = 0

    # RSI 기반 신호
    if rsi_value <= 30:
        buy_signals += 1
    elif rsi_value >= 70:
        sell_signals += 1

    # MACD 기반 신호
    if hist_value > 0 and hist.iloc[-2].item() <= 0:  # 골든크로스
        buy_signals += 1
    elif hist_value < 0 and hist.iloc[-2].item() >= 0:  # 데드크로스
        sell_signals += 1

    # 볼린저 밴드 기반 신호
    if current_price < bb_lower.iloc[-1].item():
        buy_signals += 1
    elif current_price > bb_upper.iloc[-1].item():
        sell_signals += 1

    # 이동평균 기반 신호
    if (
        ma5.iloc[-1].item() > ma20.iloc[-1].item()
        and ma5.iloc[-2].item() <= ma20.iloc[-2].item()
    ):
        buy_signals += 1
    elif (
        ma5.iloc[-1].item() < ma20.iloc[-1].item()
        and ma5.iloc[-2].item() >= ma20.iloc[-2].item()
    ):
        sell_signals += 1

    # 매매 추천
    if buy_signals > sell_signals:
        recommendation = "매수 추천"
    elif sell_signals > buy_signals:
        recommendation = "매도 추천"
    else:
        recommendation = "관망 추천"

    # 차트 생성
    create_chart(df, ma5, ma20, bb_upper, bb_lower)

    # 분석 결과 반환
    return {
        "current_price": current_price,
        "price_change": price_change,
        "price_change_percent": price_change_percent,
        "ma_status": ma_status,
        "rsi": rsi_value,
        "rsi_status": rsi_status,
        "macd": macd_value,
        "signal": signal_value,
        "macd_hist": hist_value,
        "macd_status": macd_status,
        "bollinger_upper": bb_upper.iloc[-1].item(),
        "bollinger_lower": bb_lower.iloc[-1].item(),
        "bollinger_status": bb_status,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "recommendation": recommendation,
    }


def create_chart(df, ma5, ma20, bb_upper, bb_lower):
    """
    주식 분석 차트를 생성합니다.
    """
    # result 디렉토리가 없으면 생성
    if not os.path.exists("result"):
        os.makedirs("result")

    # 차트 크기 설정
    plt.figure(figsize=(12, 8))

    # 주가 및 이동평균선
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df["close"], label="종가", color="black")
    plt.plot(df.index, ma5, label="5일 이동평균", color="red")
    plt.plot(df.index, ma20, label="20일 이동평균", color="blue")
    plt.plot(df.index, bb_upper, label="볼린저 상단", color="gray", linestyle="--")
    plt.plot(df.index, bb_lower, label="볼린저 하단", color="gray", linestyle="--")
    plt.title("주가 차트")
    plt.legend()
    plt.grid(True)

    # RSI 차트
    plt.subplot(2, 1, 2)
    rsi = calculate_rsi(df["close"])
    plt.plot(df.index[1:], rsi, label="RSI", color="purple")
    plt.axhline(y=70, color="r", linestyle="--", alpha=0.5)
    plt.axhline(y=30, color="g", linestyle="--", alpha=0.5)
    plt.title("RSI 차트")
    plt.legend()
    plt.grid(True)

    # 레이아웃 조정
    plt.tight_layout()

    # 차트 저장
    plt.savefig(f"result/stock_analysis_{df.index[-1].strftime('%Y%m%d')}.png")
    plt.close()


def main():
    # 분석할 종목 코드 입력
    stock_code = input("분석할 종목 코드를 입력하세요 (예: 005930): ")

    # 주식 기술적 분석 수행
    analyze_technical(stock_code)


if __name__ == "__main__":
    main()
