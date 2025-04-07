#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCP 환경에서 실행되는 주식 분석 시스템
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# MCP 환경 설정
MCP_API_KEY = os.getenv("MCP_API_KEY")
MCP_API_URL = os.getenv("MCP_API_URL", "https://api.mcp.example.com")
MCP_STORAGE_PATH = os.getenv("MCP_STORAGE_PATH", "/mcp/storage")

# 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"  # macOS용 한글 폰트
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지


class MCPStockAnalyzer:
    """MCP 환경에서 실행되는 주식 분석 클래스"""

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        MCP 주식 분석기 초기화

        Args:
            api_key: MCP API 키 (기본값: 환경 변수에서 로드)
            api_url: MCP API URL (기본값: 환경 변수에서 로드)
        """
        self.api_key = api_key or MCP_API_KEY
        self.api_url = api_url or MCP_API_URL

        if not self.api_key:
            logger.warning("MCP API 키가 설정되지 않았습니다.")

        # MCP 스토리지 디렉토리 생성
        os.makedirs(MCP_STORAGE_PATH, exist_ok=True)

        logger.info(f"MCP 주식 분석기 초기화 완료 (API URL: {self.api_url})")

    def fetch_stock_data(
        self, stock_code: str, period: str = "1mo"
    ) -> Optional[pd.DataFrame]:
        """
        Yahoo Finance에서 주식 데이터를 가져옵니다.

        Args:
            stock_code: 주식 코드
            period: 데이터 기간 (기본값: "1mo")

        Returns:
            주식 데이터 DataFrame 또는 None (오류 발생 시)
        """
        try:
            logger.info(f"Yahoo Finance에서 {stock_code}.KS 데이터를 가져오는 중...")
            df = yf.download(f"{stock_code}.KS", period=period)

            if df is None or df.empty:
                logger.error("데이터를 가져오는데 실패했습니다.")
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

            logger.info(f"{stock_code} 데이터 가져오기 성공: {len(df)} 행")
            return df

        except Exception as e:
            logger.error(f"주식 데이터 가져오기 오류: {e}")
            return None

    def analyze_technical(self, stock_code: str) -> Dict[str, Any]:
        """
        주식의 기술적 분석을 수행합니다.

        Args:
            stock_code: 주식 코드

        Returns:
            기술적 분석 결과 딕셔너리
        """
        # 데이터 가져오기
        df = self.fetch_stock_data(stock_code)
        if df is None or len(df) < 20:
            logger.error("충분한 데이터를 가져오지 못했습니다.")
            return {"error": "데이터 부족"}

        try:
            # 현재가 및 변동 계산
            current_price = df["close"].iloc[-1].item()
            prev_price = df["close"].iloc[-2].item()
            price_change = current_price - prev_price
            price_change_percent = (price_change / prev_price) * 100

            # 이동평균 계산
            ma5 = df["close"].rolling(window=5).mean()
            ma20 = df["close"].rolling(window=20).mean()
            ma_status = (
                "데드크로스"
                if ma5.iloc[-1].item() < ma20.iloc[-1].item()
                else "골든크로스"
            )

            # RSI 계산
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = rsi.iloc[-1].item()

            if rsi_value >= 70:
                rsi_status = "과매수"
            elif rsi_value <= 30:
                rsi_status = "과매도"
            else:
                rsi_status = "중립"

            # MACD 계산
            exp1 = df["close"].ewm(span=12, adjust=False).mean()
            exp2 = df["close"].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal

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

            # 차트 생성 및 저장
            chart_path = self.create_chart(
                df, ma5, ma20, bb_upper, bb_lower, stock_code
            )

            # 분석 결과 반환
            return {
                "current_price": current_price,
                "price_change": price_change,
                "price_change_percent": price_change_percent,
                "volume": df["volume"].iloc[-1].item(),
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
                "chart_path": chart_path,
                "technical_analysis": f"""
                현재가: {current_price:,}원
                전일대비: {price_change:+,}원 ({price_change_percent:+.2f}%)
                이동평균 상태: {ma_status}
                RSI: {rsi_value:.2f} ({rsi_status})
                MACD: {macd_value:.2f} (시그널: {signal_value:.2f}, 히스토그램: {hist_value:.2f})
                MACD 상태: {macd_status}
                볼린저 밴드 상태: {bb_status}
                매매 추천: {recommendation}
                """,
            }

        except Exception as e:
            logger.error(f"기술적 분석 중 오류 발생: {e}")
            return {"error": str(e)}

    def create_chart(
        self,
        df: pd.DataFrame,
        ma5: pd.Series,
        ma20: pd.Series,
        bb_upper: pd.Series,
        bb_lower: pd.Series,
        stock_code: str,
    ) -> str:
        """
        주식 분석 차트를 생성하고 저장합니다.

        Args:
            df: 주식 데이터 DataFrame
            ma5: 5일 이동평균
            ma20: 20일 이동평균
            bb_upper: 볼린저 상단 밴드
            bb_lower: 볼린저 하단 밴드
            stock_code: 주식 코드

        Returns:
            저장된 차트 파일 경로
        """
        try:
            # 차트 크기 설정
            plt.figure(figsize=(12, 8))

            # 주가 및 이동평균선
            plt.subplot(2, 1, 1)
            plt.plot(df.index, df["close"], label="종가", color="black")
            plt.plot(df.index, ma5, label="5일 이동평균", color="red")
            plt.plot(df.index, ma20, label="20일 이동평균", color="blue")
            plt.plot(
                df.index, bb_upper, label="볼린저 상단", color="gray", linestyle="--"
            )
            plt.plot(
                df.index, bb_lower, label="볼린저 하단", color="gray", linestyle="--"
            )
            plt.title("주가 차트")
            plt.legend()
            plt.grid(True)

            # RSI 차트
            plt.subplot(2, 1, 2)
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            plt.plot(df.index[1:], rsi, label="RSI", color="purple")
            plt.axhline(y=70, color="r", linestyle="--", alpha=0.5)
            plt.axhline(y=30, color="g", linestyle="--", alpha=0.5)
            plt.title("RSI 차트")
            plt.legend()
            plt.grid(True)

            # 레이아웃 조정
            plt.tight_layout()

            # 차트 저장
            current_date = datetime.now().strftime("%Y%m%d")
            chart_path = os.path.join(
                MCP_STORAGE_PATH, f"{stock_code}_{current_date}.png"
            )
            plt.savefig(chart_path)
            plt.close()

            logger.info(f"차트 저장 완료: {chart_path}")
            return chart_path

        except Exception as e:
            logger.error(f"차트 생성 중 오류 발생: {e}")
            return ""

    def analyze_financial(self, stock_code: str) -> Dict[str, Any]:
        """
        주식의 재무 분석을 수행합니다.

        Args:
            stock_code: 주식 코드

        Returns:
            재무 분석 결과 딕셔너리
        """
        # MCP 환경에서는 DART API 대신 MCP API를 사용
        try:
            logger.info(f"MCP API를 통해 {stock_code} 재무 데이터를 가져오는 중...")

            # MCP API 호출 (실제 구현 시 MCP API 엔드포인트로 변경)
            response = requests.get(
                f"{self.api_url}/financial/{stock_code}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(f"MCP API 오류: {response.status_code} - {response.text}")
                return {"error": f"API 오류: {response.status_code}"}

            financial_data = response.json()

            # 재무건전성 평가
            financial_health = self.evaluate_financial_health(financial_data)

            return {
                "financial_data": financial_data,
                "financial_health": financial_health,
            }

        except Exception as e:
            logger.error(f"재무 분석 중 오류 발생: {e}")
            return {"error": str(e)}

    def evaluate_financial_health(
        self, financial_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        재무건전성을 평가합니다.

        Args:
            financial_data: 재무 데이터

        Returns:
            재무건전성 평가 결과
        """
        try:
            # 실제 구현 시 재무 데이터를 기반으로 평가 로직 구현
            # 현재는 예시 데이터 반환
            return {
                "부채비율": "양호",
                "유동비율": "양호",
                "영업이익률": "양호",
                "종합 평가": "양호",
            }
        except Exception as e:
            logger.error(f"재무건전성 평가 중 오류 발생: {e}")
            return {"error": str(e)}

    def get_ai_analysis(
        self,
        stock_code: str,
        technical_data: Dict[str, Any],
        financial_data: Dict[str, Any],
    ) -> str:
        """
        AI를 활용한 종합 분석을 수행합니다.

        Args:
            stock_code: 주식 코드
            technical_data: 기술적 분석 결과
            financial_data: 재무 분석 결과

        Returns:
            AI 분석 결과
        """
        try:
            # MCP 환경에서는 OpenAI API 대신 MCP AI 서비스 사용
            logger.info(f"MCP AI 서비스를 통해 {stock_code} 종합 분석을 수행하는 중...")

            # 분석 프롬프트 구성
            prompt = f"""
            다음은 {stock_code} 종목의 종합 분석 결과입니다:

            1. 기술적 분석:
            {technical_data.get('technical_analysis', '기술적 분석 정보가 없습니다.')}

            2. 재무 분석:
            {json.dumps(financial_data.get('financial_health', {}), ensure_ascii=False, indent=2)}

            위 데이터를 바탕으로 다음 항목에 대해 상세히 분석해주세요:

            1. 종목의 전반적인 투자 매력도
            2. 수익성 분석
            3. 재무건전성 분석
            4. 성장성 분석
            5. 현금흐름 분석
            6. 기술적 분석 결과 해석
            7. 투자 위험 요소
            8. 단기/중기/장기 투자 관점에서의 투자 의견
            9. 투자 시 주의해야 할 점
            10. 종합 투자 의견
            """

            # MCP AI API 호출 (실제 구현 시 MCP AI API 엔드포인트로 변경)
            response = requests.post(
                f"{self.api_url}/ai/analyze",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"prompt": prompt},
                timeout=60,
            )

            if response.status_code != 200:
                logger.error(
                    f"MCP AI API 오류: {response.status_code} - {response.text}"
                )
                return f"AI 분석 중 오류 발생: {response.status_code}"

            result = response.json()
            return result.get("analysis", "AI 분석 결과를 가져오지 못했습니다.")

        except Exception as e:
            logger.error(f"AI 분석 중 오류 발생: {e}")
            return f"AI 분석 중 오류 발생: {e}"

    def save_analysis_result(
        self,
        stock_code: str,
        company_name: str,
        technical_data: Dict[str, Any],
        financial_data: Dict[str, Any],
        ai_analysis: str,
    ) -> str:
        """
        분석 결과를 저장합니다.

        Args:
            stock_code: 주식 코드
            company_name: 회사명
            technical_data: 기술적 분석 결과
            financial_data: 재무 분석 결과
            ai_analysis: AI 분석 결과

        Returns:
            저장된 결과 파일 경로
        """
        try:
            # 결과 저장
            current_date = datetime.now().strftime("%Y%m%d")
            result_file = os.path.join(
                MCP_STORAGE_PATH, f"{company_name}({stock_code})_{current_date}.txt"
            )

            with open(result_file, "w", encoding="utf-8") as f:
                f.write(f"=== {company_name}({stock_code}) 종목 분석 결과 ===\n\n")

                # 기술적 분석 결과
                f.write("1. 기술적 분석\n")
                f.write("-" * 50 + "\n")
                f.write(
                    technical_data.get(
                        "technical_analysis", "기술적 분석 정보가 없습니다."
                    )
                )
                f.write("\n\n")

                # 재무제표 분석 결과
                f.write("2. 재무제표 분석\n")
                f.write("-" * 50 + "\n")
                financial = financial_data.get("financial_data", {})
                f.write(json.dumps(financial, ensure_ascii=False, indent=2))
                f.write("\n\n")

                # 재무건전성 평가
                f.write("3. 재무건전성 평가\n")
                f.write("-" * 50 + "\n")
                f.write(
                    json.dumps(
                        financial_data.get("financial_health", {}),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                f.write("\n\n")

                # 종합 평가
                f.write("4. 종합 평가\n")
                f.write("-" * 50 + "\n")
                f.write(f"기술적 분석: {technical_data.get('recommendation', 'N/A')}\n")
                f.write(
                    f"재무건전성: {financial_data.get('financial_health', {}).get('종합 평가', 'N/A')}\n\n"
                )

                # AI 분석 결과
                f.write("5. AI 분석 결과\n")
                f.write("-" * 50 + "\n")
                f.write(ai_analysis)

            logger.info(f"분석 결과 저장 완료: {result_file}")
            return result_file

        except Exception as e:
            logger.error(f"분석 결과 저장 중 오류 발생: {e}")
            return ""

    def analyze_stock(self, stock_code: str) -> Dict[str, Any]:
        """
        주식 종합 분석을 수행합니다.

        Args:
            stock_code: 주식 코드

        Returns:
            종합 분석 결과
        """
        try:
            logger.info(f"{stock_code} 종목 분석 시작...")

            # 회사명 가져오기 (MCP API 사용)
            company_name = self.get_company_name(stock_code)

            # 기술적 분석
            technical_data = self.analyze_technical(stock_code)
            if "error" in technical_data:
                logger.error(f"기술적 분석 실패: {technical_data['error']}")
                return {"error": f"기술적 분석 실패: {technical_data['error']}"}

            # 재무 분석
            financial_data = self.analyze_financial(stock_code)
            if "error" in financial_data:
                logger.error(f"재무 분석 실패: {financial_data['error']}")
                return {"error": f"재무 분석 실패: {financial_data['error']}"}

            # AI 분석
            ai_analysis = self.get_ai_analysis(
                stock_code, technical_data, financial_data
            )

            # 결과 저장
            result_file = self.save_analysis_result(
                stock_code, company_name, technical_data, financial_data, ai_analysis
            )

            return {
                "stock_code": stock_code,
                "company_name": company_name,
                "technical_analysis": technical_data,
                "financial_analysis": financial_data,
                "ai_analysis": ai_analysis,
                "result_file": result_file,
            }

        except Exception as e:
            logger.error(f"종목 분석 중 오류 발생: {e}")
            return {"error": str(e)}

    def get_company_name(self, stock_code: str) -> str:
        """
        주식 코드로 회사명을 가져옵니다.

        Args:
            stock_code: 주식 코드

        Returns:
            회사명
        """
        try:
            # MCP API 호출 (실제 구현 시 MCP API 엔드포인트로 변경)
            response = requests.get(
                f"{self.api_url}/company/{stock_code}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"회사명 조회 실패: {response.status_code} - {response.text}"
                )
                return stock_code

            result = response.json()
            return result.get("company_name", stock_code)

        except Exception as e:
            logger.error(f"회사명 조회 중 오류 발생: {e}")
            return stock_code


def main():
    """메인 함수"""
    try:
        # MCP 주식 분석기 초기화
        analyzer = MCPStockAnalyzer()

        # 사용자 입력 받기
        stock_code = input("분석할 주식 코드를 입력하세요: ")

        # 종목 분석 수행
        result = analyzer.analyze_stock(stock_code)

        if "error" in result:
            print(f"분석 중 오류 발생: {result['error']}")
            return

        # 결과 출력
        print(
            f"\n===== {result['company_name']}({result['stock_code']}) 종목 분석 결과 ====="
        )
        print(f"기술적 분석: {result['technical_analysis']['recommendation']}")
        print(
            f"재무건전성: {result['financial_analysis']['financial_health']['종합 평가']}"
        )
        print(f"\n분석 결과가 {result['result_file']}에 저장되었습니다.")

    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
