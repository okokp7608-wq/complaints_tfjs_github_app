# 민원 샘플링 TensorFlow.js 웹 서빙 앱

이 폴더는 `complaints.parquet` 원천 데이터에서 **priority 클래스별 균형 샘플링 데이터만 추출**해 만든 정적 웹앱입니다. GitHub Pages에 그대로 올릴 수 있으며, 브라우저에서 TensorFlow.js 모델을 학습하고 예측 서빙 현황을 확인할 수 있습니다.

## 구성 파일

```text
index.html                 # 웹 UI
styles.css                 # 화면 스타일
app.js                     # TensorFlow.js 학습·예측 로직
sampled_complaints.json    # 클래스별 균형 샘플링 데이터
metadata.json              # 원천/샘플 메타데이터
README.md                  # 실행 안내
```

## 실행 방법

로컬에서 `file://`로 직접 열면 브라우저 보안 정책 때문에 JSON 파일을 읽지 못할 수 있습니다. 아래처럼 간단한 서버로 실행하세요.

```bash
cd complaints_tfjs_github_app
python -m http.server 8000
```

브라우저에서 아래 주소 접속:

```text
http://localhost:8000
```

## GitHub Pages 배포

1. GitHub 저장소 생성
2. 이 폴더 안의 파일을 저장소 루트에 업로드
3. Settings → Pages → Branch를 `main` / root로 지정
4. 배포 URL 접속

## 모델 방식

- 입력 변수: `channel`, `complaint_type`, `region_code`, `staff_career`, `process_days`, `internal_consult`, `escalated`, `text_length`, `attachment_count`, `satisfaction`, `resubmit`, `budget_impact`
- 목표 변수: `priority`
- 전처리: 숫자형 표준화 + 범주형 원-핫 인코딩
- 모델: TensorFlow.js Sequential Dense Neural Network
- 저장: 브라우저 IndexedDB

## 주의사항

이 웹앱은 학습·서빙 구조 증적과 데모를 위한 프론트엔드 예시입니다. 운영 환경에서는 서버 학습 모델의 검증, 모델 버전관리, 재현성 로그, 개인정보 비식별 처리, 접근통제, 성능 모니터링이 추가로 필요합니다.
