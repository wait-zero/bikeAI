#!/usr/bin/env bash
# Manual download instructions for the static datasets we need.
# Most Seoul open-data files are gated behind a click-through, so we list URLs
# rather than wget them blindly.
set -euo pipefail

cat <<'EOF'
=== bikeAI: 정적 데이터 다운로드 안내 ===

다음 파일들을 data/raw/ 아래에 받아주세요.

1) 따릉이 대여이력 (월별 CSV/XLSX, 가장 큰 데이터)
   URL: https://data.seoul.go.kr/dataList/OA-15182/F/1/datasetView.do
   최근 1~2년치만 받아도 충분히 학습 가능합니다.
   저장: data/raw/rental_history/<year>/<month>.csv

2) 따릉이 대여소 마스터
   URL: https://www.data.go.kr/data/15099365/fileData.do
   저장: data/raw/stations.csv

3) 기상청 ASOS 시간 자료 (서울 108)
   URL: https://data.kma.go.kr  →  자료 → 종관기상관측(ASOS) → 시간자료
   기간을 따릉이와 동일하게 맞춰서 CSV 다운로드
   저장: data/raw/asos_seoul.csv

4) 실시간 API 인증키
   URL: https://www.data.go.kr/data/15126639/openapi.do
   발급 후 환경변수에 설정:
       export PUBLIC_BIKE_API_KEY="..."

다운로드가 끝나면 다음 단계로 진행:
    python scripts/02_reconstruct_timeseries.py
    python scripts/03_build_features.py
    python scripts/04_train_lgbm.py --train-end 2025-07-01 --valid-end 2025-10-01
EOF
