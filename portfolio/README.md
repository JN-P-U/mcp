# MCP 주식 분석 시스템

MCP(Multi-Cloud Platform) 환경에서 실행되는 주식 분석 시스템입니다. 이 시스템은 기술적 분석, 재무 분석, AI 기반 종합 분석을 제공합니다.

## 기능

- **기술적 분석**: 이동평균, RSI, MACD, 볼린저 밴드 등을 활용한 기술적 분석
- **재무 분석**: 재무제표 분석 및 재무건전성 평가
- **AI 기반 종합 분석**: 기술적 분석과 재무 분석 결과를 바탕으로 한 AI 기반 종합 분석
- **차트 생성**: 주가 차트 및 기술적 지표 차트 생성

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/mcp-stock-analysis.git
cd mcp-stock-analysis
```

2. 의존성 설치
```bash
poetry install
```

3. 환경 변수 설정
`.env` 파일을 생성하고 다음 환경 변수를 설정합니다:
```
MCP_API_KEY=your_mcp_api_key_here
MCP_API_URL=https://api.mcp.example.com
MCP_STORAGE_PATH=/mcp/storage
```

## 사용 방법

1. 프로그램 실행
```bash
poetry run python main.py
```

2. 주식 코드 입력
프롬프트가 표시되면 분석할 주식 코드를 입력합니다.

3. 분석 결과 확인
분석 결과는 콘솔에 출력되며, 상세 결과는 `MCP_STORAGE_PATH` 디렉토리에 저장됩니다.

## MCP 환경 설정

이 시스템은 MCP 환경에서 실행되도록 설계되었습니다. MCP 환경에서는 다음과 같은 API를 사용합니다:

- **회사 정보 API**: 주식 코드로 회사명을 조회
- **재무 데이터 API**: 주식 코드로 재무 데이터를 조회
- **AI 분석 API**: 기술적 분석과 재무 분석 결과를 바탕으로 AI 기반 종합 분석 수행

## 라이선스

MIT
