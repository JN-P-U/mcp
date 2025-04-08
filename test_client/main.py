import asyncio
import json
import os
import traceback

import httpx
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 키 확인
openai_api_key = os.getenv("OPENAI_API_KEY")
dart_api_key = os.getenv("DART_API_KEY")
mcp_api_key = os.getenv("MCP_API_KEY")

if not all([openai_api_key, dart_api_key, mcp_api_key]):
    raise ValueError("필요한 API 키가 모두 설정되지 않았습니다.")


async def main():
    try:
        print("API 요청 초기화 중...")

        # API 엔드포인트 및 헤더 설정
        url = os.getenv("MCP_API_URL")
        headers = {
            "Authorization": f"Bearer {mcp_api_key}",
            "Content-Type": "application/json",
        }

        # API 요청 데이터
        data = {
            "stock_code": "005930",  # 삼성전자
            "openai_api_key": openai_api_key,
            "dart_api_key": dart_api_key,
            "mcp_api_key": mcp_api_key,
        }

        print("\nAPI 요청 데이터:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        print("\nAPI 요청 실행 중...")
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=data, headers=headers)

            print(f"\n응답 상태 코드: {response.status_code}")
            print(
                "응답 헤더:",
                json.dumps(dict(response.headers), indent=2, ensure_ascii=False),
            )

            if response.status_code == 404:
                print("\nAPI 엔드포인트를 찾을 수 없습니다.")
                print("URL:", url)
            elif response.status_code == 401:
                print("\n인증에 실패했습니다. API 키를 확인해주세요.")
            elif response.status_code != 200:
                print(f"\nAPI 요청 실패 (상태 코드: {response.status_code})")

            print("\n응답 내용:")
            try:
                response_json = response.json()
                print(json.dumps(response_json, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(response.text)

    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        print(f"오류 타입: {type(e).__name__}")
        print("\n상세 오류 정보:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
