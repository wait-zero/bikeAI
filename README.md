# bikeAI — 서울 따릉이 가용 자전거 예측 (MVP)

대여소별 자전거 대수를 5/15/30/60분 후까지 예측하는 모델 학습/평가 파이프라인.

## 빠른 시작

```bash
# 1) 환경 설치 (uv 권장)
uv sync           # 또는: pip install -e .[dev]

# 2) 환경변수
export PUBLIC_BIKE_API_KEY="<data.go.kr 15126639 인증키>"

# 3) 단위 테스트
pytest

# 4) 실시간 수집 (수동 1회 호출)
python -m bikeai.ingest.realtime --once

# 5) 정적 데이터 다운로드 안내
bash scripts/01_download_data.sh
```

자세한 설계는 `docs/` 또는 `/home/hidi/.claude/plans/jaunty-wobbling-bachman.md` 참고.

## 데이터 출처

- 따릉이 대여이력: 서울 열린데이터광장 OA-15182 (공공누리 1유형)
- 따릉이 대여소 마스터: data.go.kr 15099365
- 실시간 API: data.go.kr 15126639
- 기상청 ASOS(서울 108): data.kma.go.kr
