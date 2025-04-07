# 종목코드 다운로드

import numpy as np


def calculate_technical_indicators(df):
    # 추가: 이동평균선 상태
    df["MA_Status"] = np.where(df["MA5"] > df["MA20"], "골든크로스", "데드크로스")

    # 추가: MACD 상태
    df["MACD_Status"] = np.where(df["MACD"] > df["Signal"], "상승", "하락")

    # 추가: 볼린저 밴드 상태 및 Upper, Lower 별칭
    df["Upper"] = df["BB_Upper"]
    df["Lower"] = df["BB_Lower"]
    df["BB_Status"] = np.where(
        df["close"] > df["BB_Upper"],
        "과매수",
        np.where(df["close"] < df["BB_Lower"], "과매도", "보통"),
    )


if __name__ == "__main__":
    main()
