# BE (wait-zero) ↔ AI 서버 (bikeAI) 연동 가이드

**작성일**: 2026-04-12
**대상 레포**: `wait-zero/backend`
**AI 서버 주소**: `http://ai.waitzero.site:8000`
**AI 서버 API Key**: `test-key-bikeai-2026` (프로덕션 전 교체)

---

## 1. 개요

AI 서버는 이미 완성·배포되어 아래 형식으로 동작 중입니다. BE 의 `AiForecastClient` 가 이 형식에 맞추면 연동 완료.

### AI 서버 `/predict` 스펙

**Request:**
```http
POST http://ai.waitzero.site:8000/predict
Content-Type: application/json
X-API-Key: test-key-bikeai-2026

{
  "station_id": "ST-10",
  "current_bike_count": 8,
  "horizons": [5, 15, 30, 60],
  "history": [
    {"minutes_ago": 1, "bike_count": 8},
    {"minutes_ago": 5, "bike_count": 9},
    {"minutes_ago": 15, "bike_count": 11}
  ],
  "weather": {"temp_c": 12.5, "precip_mm": 0}
}
```

- `station_id` (필수): `"ST-10"` 형식 = Station.externalId
- `current_bike_count` (필수): 현재 자전거 수 = StationSnapshot.availableBikes
- `horizons` (선택): 기본 `[5, 15, 30, 60]`
- `history` (선택): 정확도 향상. 없으면 degraded 모드.
- `weather` (선택): 없으면 NaN 처리됨.

**Response:**
```json
{
  "station_id": "ST-10",
  "station_name": "서울특별시 마포구 양화로 93",
  "lat": 37.552746,
  "lon": 126.918617,
  "as_of": "2026-04-12T04:43:01.854626+00:00",
  "current_bike_count": 8,
  "predictions": [
    {"horizon_min": 5,  "predicted_count": 8,  "predicted_delta": 0.0,  "method": "persistence"},
    {"horizon_min": 15, "predicted_count": 8,  "predicted_delta": 0.05, "method": "model"},
    {"horizon_min": 30, "predicted_count": 9,  "predicted_delta": 0.53, "method": "model"},
    {"horizon_min": 60, "predicted_count": 13, "predicted_delta": 4.63, "method": "model"}
  ],
  "warnings": [
    "No history provided; lag features filled with current count (degraded)"
  ],
  "model_version": "v1"
}
```

### 기타 엔드포인트 (참고)
- `GET /health` — 인증 불필요, 서버 상태 확인
- `GET /stations?limit=50` — 인증 필요, 대여소 목록
- `POST /predict/batch` — 인증 필요, 여러 대여소 한번에
- `GET /docs` — Swagger UI (브라우저에서 직접 테스트)

---

## 2. 변경할 파일 목록 (4개 + 테스트 1개)

| # | 파일 | 변경 유형 |
|---|---|---|
| 1 | `application.yml` | 설정 추가 (api-key) |
| 2 | `ForecastPort.java` | 인터페이스 시그니처 변경 |
| 3 | `ForecastService.java` | StationRepository 주입 + 조회 로직 |
| 4 | `AiForecastClient.java` | AI 서버 형식에 맞춰 전면 교체 |
| 5 | `WaitZeroApplicationTests.java` | mock 시그니처 수정 |

**변경 불필요**: StationController, NotificationScheduler, StationDtos, ForecastResult — 모두 `ForecastService.getForecast(Long, int)` 만 사용하므로 영향 없음.

---

## 3. 변경 상세

### 3-1. `application.yml` — api-key 추가

**파일**: `src/main/resources/application.yml`

**변경 전 (44행 부근):**
```yaml
waitzero:
  ai:
    base-url: "http://ai:8000"
    timeout-ms: 3000
```

**변경 후:**
```yaml
waitzero:
  ai:
    base-url: "http://ai:8000"
    api-key: ${BIKEAI_API_KEY:test-key-bikeai-2026}
    timeout-ms: 3000
```

**`application-docker.yml` 도 동일하게:**
```yaml
waitzero:
  ai:
    base-url: ${WAITZERO_AI_BASE_URL:http://ai:8000}
    api-key: ${BIKEAI_API_KEY:test-key-bikeai-2026}
```

**`application-prod.yml`:**
```yaml
waitzero:
  ai:
    base-url: ${WAITZERO_AI_BASE_URL}
    api-key: ${BIKEAI_API_KEY}
```

**`application-dev.yml` (로컬 개발용):**
```yaml
waitzero:
  ai:
    base-url: "http://localhost:8000"
    api-key: "test-key-bikeai-2026"
```

---

### 3-2. `ForecastPort.java` — 파라미터 추가

**파일**: `src/main/java/kr/waitzero/bike/application/forecast/ForecastPort.java`

**변경 전:**
```java
ForecastResult forecast(Long stationId, int horizonMinutes);
```

**변경 후:**
```java
ForecastResult forecast(Long stationId, String externalId,
                        int currentBikeCount, int horizonMinutes);
```

**전체 파일:**
```java
package kr.waitzero.bike.application.forecast;

import java.time.Instant;
import java.util.List;

public interface ForecastPort {

    ForecastResult forecast(Long stationId, String externalId,
                            int currentBikeCount, int horizonMinutes);

    record ForecastResult(
            Long stationId,
            Instant generatedAt,
            List<Point> points,
            Instant depletionAt,
            String confidence,
            String reason,
            String model
    ) {
        public record Point(Instant at, int expectedBikes) {}
    }
}
```

---

### 3-3. `ForecastService.java` — StationRepository 주입 + externalId/currentBikes 조회

**파일**: `src/main/java/kr/waitzero/bike/application/forecast/ForecastService.java`

**전체 파일 (교체):**
```java
package kr.waitzero.bike.application.forecast;

import kr.waitzero.bike.domain.station.Station;
import kr.waitzero.bike.domain.station.StationRepository;
import kr.waitzero.bike.domain.station.StationSnapshot;
import kr.waitzero.bike.domain.station.StationSnapshotRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
@Service
@RequiredArgsConstructor
public class ForecastService {

    private static final Duration FRESH_SNAPSHOT_WINDOW = Duration.ofMinutes(30);
    private static final Duration CACHE_TTL = Duration.ofSeconds(60);
    private static final String CONFIDENCE_NONE = "none";
    private static final String REASON_INSUFFICIENT_HISTORY = "insufficient_history";

    private final ForecastPort forecastPort;
    private final StationSnapshotRepository snapshotRepository;
    private final StationRepository stationRepository;   // ← 추가

    private final ConcurrentHashMap<CacheKey, CacheEntry> cache = new ConcurrentHashMap<>();

    public ForecastPort.ForecastResult getForecast(Long stationId, int horizonMinutes) {
        int clamped = Math.min(Math.max(horizonMinutes, 5), 120);

        CacheKey key = new CacheKey(stationId, clamped);
        Instant now = Instant.now();
        CacheEntry hit = cache.get(key);
        if (hit != null && hit.expiresAt.isAfter(now)) {
            return hit.result;
        }

        // 최근 스냅샷 확인
        Optional<StationSnapshot> latest =
                snapshotRepository.findTopByStationIdOrderByObservedAtDesc(stationId);
        if (latest.isEmpty() || !isRecent(latest.get(), now)) {
            log.debug("forecast skipped: no recent snapshot stationId={}", stationId);
            ForecastPort.ForecastResult stale = insufficientHistory(stationId, now);
            cache.put(key, new CacheEntry(stale, now.plus(CACHE_TTL)));
            return stale;
        }

        // ── 추가: externalId + currentBikeCount 조회 ──
        Station station = stationRepository.findById(stationId).orElse(null);
        if (station == null) {
            log.warn("forecast skipped: station not found stationId={}", stationId);
            return insufficientHistory(stationId, now);
        }
        String externalId = station.getExternalId();
        int currentBikes = latest.get().getAvailableBikes();
        // ───────────────────────────────────────────────

        ForecastPort.ForecastResult result =
                forecastPort.forecast(stationId, externalId, currentBikes, clamped);
        cache.put(key, new CacheEntry(result, now.plus(CACHE_TTL)));
        return result;
    }

    private boolean isRecent(StationSnapshot snapshot, Instant now) {
        return snapshot.getObservedAt() != null
                && snapshot.getObservedAt().isAfter(now.minus(FRESH_SNAPSHOT_WINDOW));
    }

    private static ForecastPort.ForecastResult insufficientHistory(Long stationId, Instant now) {
        return new ForecastPort.ForecastResult(
                stationId, now, List.of(), null,
                CONFIDENCE_NONE, REASON_INSUFFICIENT_HISTORY, null
        );
    }

    private record CacheKey(Long stationId, int horizonMinutes) {}
    private record CacheEntry(ForecastPort.ForecastResult result, Instant expiresAt) {}
}
```

---

### 3-4. `AiForecastClient.java` — AI 서버 형식에 맞춰 전면 교체

**파일**: `src/main/java/kr/waitzero/bike/infrastructure/ai/AiForecastClient.java`

**전체 파일 (교체):**
```java
package kr.waitzero.bike.infrastructure.ai;

import kr.waitzero.bike.application.forecast.ForecastPort;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.time.Instant;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * bikeAI FastAPI 서버 HTTP 클라이언트.
 *
 * AI 서버 요청:
 *   POST /predict
 *   X-API-Key: {api-key}
 *   {"station_id":"ST-10", "current_bike_count":8, "horizons":[5,15,30,60]}
 *
 * AI 서버 응답:
 *   {"predictions":[{"horizon_min":5,"predicted_count":8,"predicted_delta":0.0,"method":"persistence"},...],
 *    "as_of":"...", "model_version":"v1", "warnings":[...]}
 *
 * 이 클라이언트는 AI 응답 → ForecastResult 내부 모델로 변환하는 adapter.
 */
@Slf4j
@Component
public class AiForecastClient implements ForecastPort {

    private static final String CONFIDENCE_NONE = "none";
    private static final String CONFIDENCE_MEDIUM = "medium";
    private static final String CONFIDENCE_LOW = "low";
    private static final String REASON_UNAVAILABLE = "ai_unavailable";

    private final RestClient restClient;

    public AiForecastClient(
            @Value("${waitzero.ai.base-url}") String baseUrl,
            @Value("${waitzero.ai.api-key:}") String apiKey
    ) {
        var builder = RestClient.builder().baseUrl(baseUrl);
        if (apiKey != null && !apiKey.isBlank()) {
            builder.defaultHeader("X-API-Key", apiKey);
        }
        builder.defaultHeader("Content-Type", "application/json");
        this.restClient = builder.build();
    }

    @Override
    public ForecastResult forecast(Long stationId, String externalId,
                                   int currentBikeCount, int horizonMinutes) {
        try {
            @SuppressWarnings("unchecked")
            Map<String, Object> resp = restClient.post()
                    .uri("/predict")
                    .body(Map.of(
                            "station_id", externalId,
                            "current_bike_count", currentBikeCount,
                            "horizons", List.of(5, 15, 30, 60)
                    ))
                    .retrieve()
                    .body(Map.class);

            if (resp == null) {
                log.warn("AI forecast empty body stationId={} externalId={}", stationId, externalId);
                return fallback(stationId);
            }

            return mapResponse(stationId, resp);
        } catch (Exception e) {
            log.warn("AI forecast call failed stationId={} externalId={} err={}",
                    stationId, externalId, e.toString());
            return fallback(stationId);
        }
    }

    // ── AI 응답 → ForecastResult 변환 ──────────────────────────────

    private ForecastResult mapResponse(Long stationId, Map<String, Object> resp) {
        Instant generatedAt = parseInstantOrNow(resp.get("as_of"));
        String model = asString(resp.get("model_version"));

        // warnings 유무로 confidence 결정
        List<?> warnings = resp.get("warnings") instanceof List<?> w ? w : List.of();
        String confidence = warnings.isEmpty() ? CONFIDENCE_MEDIUM : CONFIDENCE_LOW;

        // predictions → points 변환
        List<ForecastResult.Point> points = new ArrayList<>();
        Instant depletionAt = null;

        Object rawPredictions = resp.get("predictions");
        if (rawPredictions instanceof List<?> list) {
            for (Object item : list) {
                if (!(item instanceof Map<?, ?> pm)) continue;
                Integer horizonMin = asInt(pm.get("horizon_min"));
                Integer predictedCount = asInt(pm.get("predicted_count"));
                if (horizonMin == null || predictedCount == null) continue;

                // horizon 분 → 절대 시각
                Instant at = generatedAt.plusSeconds(horizonMin * 60L);
                points.add(new ForecastResult.Point(at, predictedCount));

                // 처음으로 0대가 되는 시점 = depletionAt
                if (predictedCount == 0 && depletionAt == null) {
                    depletionAt = at;
                }
            }
        }

        return new ForecastResult(
                stationId,
                generatedAt,
                points,
                depletionAt,
                confidence,
                null,   // reason
                model
        );
    }

    // ── fallback (네트워크 실패 시) ─────────────────────────────────

    private static ForecastResult fallback(Long stationId) {
        return new ForecastResult(
                stationId, Instant.now(), List.of(), null,
                CONFIDENCE_NONE, REASON_UNAVAILABLE, null
        );
    }

    // ── util ────────────────────────────────────────────────────────

    private static String asString(Object v) {
        return v == null ? null : v.toString();
    }

    private static Integer asInt(Object v) {
        if (v instanceof Number n) return n.intValue();
        if (v instanceof String s) {
            try { return Integer.parseInt(s); } catch (NumberFormatException ignored) {}
        }
        return null;
    }

    private static Instant parseInstantOrNow(Object v) {
        if (v == null) return Instant.now();
        String s = v.toString();
        try {
            return Instant.parse(s);
        } catch (Exception ignore) {
            try {
                return OffsetDateTime.parse(s).toInstant();
            } catch (Exception ignore2) {
                return Instant.now();
            }
        }
    }
}
```

---

### 3-5. `WaitZeroApplicationTests.java` — mock 시그니처 수정

**파일**: `src/test/java/kr/waitzero/bike/WaitZeroApplicationTests.java`

**변경 전 (158~159행):**
```java
Mockito.when(forecastPort.forecast(Mockito.eq(s.getId()), Mockito.anyInt()))
        .thenReturn(new ForecastPort.ForecastResult(
```

**변경 후:**
```java
Mockito.when(forecastPort.forecast(
                Mockito.eq(s.getId()),
                Mockito.anyString(),        // externalId
                Mockito.anyInt(),            // currentBikeCount
                Mockito.anyInt()             // horizonMinutes
        ))
        .thenReturn(new ForecastPort.ForecastResult(
```

나머지는 동일. `ForecastResult` 구조는 변경 없음.

---

## 4. 변환 매핑 참조

```
[AI 서버 응답]                              [BE ForecastResult]
──────────────                              ──────────────────
as_of                                  →    generatedAt
model_version                          →    model
predictions[].horizon_min              →    points[].at  (= generatedAt + horizon분)
predictions[].predicted_count          →    points[].expectedBikes
predictions[0 where predicted_count=0] →    depletionAt (처음 0대 되는 시점, 없으면 null)
warnings 비어있으면                     →    confidence = "medium"
warnings 있으면                         →    confidence = "low"
에러/타임아웃                           →    confidence = "none", reason = "ai_unavailable"
```

---

## 5. 배포 환경 변수

| 환경 변수 | 값 | 설명 |
|---|---|---|
| `WAITZERO_AI_BASE_URL` | `http://ai.waitzero.site:8000` | AI 서버 주소 |
| `BIKEAI_API_KEY` | `test-key-bikeai-2026` | AI 서버 인증 키 (프로덕션 전 교체) |

Docker Compose 라면:
```yaml
services:
  backend:
    environment:
      WAITZERO_AI_BASE_URL: "http://ai.waitzero.site:8000"
      BIKEAI_API_KEY: "test-key-bikeai-2026"
```

---

## 6. 검증 방법

### 단계 1: AI 서버 상태 확인
```bash
curl http://ai.waitzero.site:8000/health
# → {"status":"ok","model_loaded":true,"stations":2735}
```

### 단계 2: AI 직접 호출 (BE 없이)
```bash
curl -X POST http://ai.waitzero.site:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-key-bikeai-2026" \
  -d '{"station_id":"ST-10","current_bike_count":8}'
```

### 단계 3: BE 테스트 실행
```bash
cd backend
./gradlew test
```

### 단계 4: BE 에서 forecast API 호출
```bash
# BE 서버가 떠 있을 때
curl http://localhost:8080/api/v1/stations/{id}/forecast?minutes=30
```

이 응답에 `confidence: "medium"` 과 `points` 배열이 나오면 연동 성공.

---

## 7. 주의사항

### AI 서버 제한
- **학습 데이터**: 2026-04-11 오후(13:39~16:24)만. 출퇴근 시간 정확도 보장 안 됨.
- **5분 예측**: 항상 persistence (변화 없음). 자연 한계.
- **API 서버 장애 시**: `AiForecastClient` 가 자동 fallback (confidence=none, reason=ai_unavailable).

### station_id 매핑
- AI 는 `"ST-10"` (= Station.externalId) 사용
- BE 내부는 `Long id` (DB PK) 사용
- 변환: ForecastService 에서 Station 조회 → externalId 추출 → AI 에 전달

### API Key 관리
- `test-key-bikeai-2026` 는 테스트용. 프로덕션 전 교체 필수.
- 교체 방법: AI 서버 데스크탑에서 `BIKEAI_API_KEY` 환경변수 변경 후 재시작. BE 에서도 동일한 키를 환경변수로 설정.

---

## 8. 데이터 흐름 (최종)

```
Flutter 앱
    ↓ GET /api/v1/stations/{id}/forecast?minutes=30
    ↓
Spring Boot BE (Oracle 1GB)
    ↓ StationController.forecast()
    ↓ ForecastService.getForecast(stationId, 30)
    ↓   1) 캐시 확인 (60초 TTL)
    ↓   2) 최근 스냅샷 확인 (30분 이내)
    ↓   3) Station 조회 → externalId, currentBikes
    ↓
    ↓ AiForecastClient.forecast(id, "ST-10", 8, 30)
    ↓   POST http://ai.waitzero.site:8000/predict
    ↓   X-API-Key: test-key-bikeai-2026
    ↓   {"station_id":"ST-10","current_bike_count":8}
    ↓
데스크탑 AI 서버
    ↓ LightGBM 추론 (~5ms)
    ↓ 응답: predictions[{5min:8, 15min:8, 30min:9, 60min:13}]
    ↓
Spring Boot BE
    ↓ AiForecastClient.mapResponse()
    ↓ → ForecastResult{points, confidence, depletionAt, ...}
    ↓
Flutter 앱
    ↓ 사용자에게 표시:
    ↓ "30분 후 자전거 9대 예상"
```
