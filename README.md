# 멀티모달 기반 강화학습 트레이딩 시스템

## 프로젝트 구조
```
paper1/
├── data/                      # 데이터 저장 디렉토리
│   ├── raw/                   # 원본 데이터
│   │   ├── train/            # 학습 데이터
│   │   │   ├── 1h/          # 1시간봉 데이터
│   │   │   ├── 4h/          # 4시간봉 데이터
│   │   │   └── 1d/          # 일봉 데이터
│   │   └── test/            # 테스트 데이터
│   │       ├── 1h/          # 1시간봉 데이터
│   │       ├── 4h/          # 4시간봉 데이터
│   │       └── 1d/          # 일봉 데이터
│   └── processed/            # 전처리된 데이터
├── models/                    # 모델 저장 디렉토리
│   ├── checkpoints/          # 학습 체크포인트
│   └── saved_models/         # 저장된 모델
├── results/                   # 결과 저장 디렉토리
│   ├── performance/          # 성능 측정 결과
│   └── images/               # 생성된 시각화 이미지
├── advanced_multimodal_trader.py   # 멀티모달 트레이딩 시스템의 핵심 구현
├── agent_ensemble.py          # 에이전트 앙상블 구현
├── trading_strategy.py        # 트레이딩 전략 구현
├── multimodal_fusion.py       # 멀티모달 데이터 융합 구현
├── basic_trader.py            # 기본 트레이더 구현
├── candlestick_analyzer.py    # 캔들스틱 패턴 분석기
├── sentiment_analyzer.py      # 감성 분석기
├── ensemble_trader.py         # 앙상블 트레이더 구현
├── run_paper1_multimodal_test.py  # 실행 스크립트
├── run_advanced_multimodal_trader.py  # 실행 스크립트
├── generate_performance_graphs.py  # 그래프 생성
├── generate_research_graphs.py     # 연구 그래프 생성
├── backtesting.py             # 백테스팅 구현
├── feature_fusion.py          # 특성 융합 구현
├── requirements.txt           # 프로젝트 의존성
└── README.md                  # 프로젝트 문서
```

## 시스템 아키텍처

### 1. 데이터 소스 레이어
#### 캔들스틱 데이터 소스
- 바이낸스/트레이딩뷰에서 제공하는 OHLCV(시가, 고가, 저가, 종가, 거래량) 시계열 데이터
- 다양한 시간대(1H, 4H, 1D)의 가격 패턴 데이터 포함
- 실시간 및 과거 시장 데이터의 통합 소스

#### 뉴스 기사 데이터 소스
- 알고리즘 관련 뉴스 API 및 RSS 피드에서 수집
- 시장 감성 및 이벤트 정보 제공
- 금융 뉴스, 소셜 미디어 감성 분석, 공식 발표 등 포함

### 2. 이미지 기반 분석 모듈
#### 캔들스틱 데이터를 시각적 패턴으로 처리하는 딥러닝 모듈:
- **캔들스틱 차트 생성 [224,224,3]**
  - 시계열 데이터를 224x224 픽셀 크기의 RGB 이미지로 변환
  - API/간거래/가격 변화를 시각적으로 표현하여 패턴 인식에 최적화
- **Attention-based CNN**
  - 컨볼루션 신경망(CNN)과 어텐션 메커니즘을 결합
  - 중요한 시각적 패턴(지지선, 저항선, 헤드앤숄더 등)에 집중
  - 복잡한 차트 패턴을 딥러닝으로 자동 인식
- **포지션 및 펀딩 레이트 특징**
  - CBAM(Convolutional Block Attention Module) 메커니즘 적용
  - 현재 시장 상태와 거래 비용을 고려한 특징 추출
  - 중요 시각 영역에 가중치 부여로 노이즈 감소
- **[512,2,1] 벡터 출력**
  - 차트 패턴을 512 차원의 특징 벡터로 압축
  - 다음 단계의 멀티모달 융합을 위한 최적 표현 생성

### 3. DeepSeek-R1 뉴스 분석 모듈
#### 뉴스 텍스트 데이터를 분석하는 대규모 언어 모델 기반 모듈:
- **뉴스 데이터 전처리**
  - 토큰화, 불용어 제거, 핵심 추출 등의 NLP 전처리 작업
  - 금융 도메인 특화 용어 처리 및 정보 구조화
- **DeepSeek-R1 32B 모델**
  - 320억 파라미터 규모의 대형 언어 모델
  - Chain-of-Thought 추론 방식으로 복잡한 금융 정보 분석
  - 여러 단계의 사고 과정을 통해 뉴스의 시장 영향 평가
- **시장 영향 점수 분석 (-1~1)**
  - 뉴스가 시장에 미치는 영향을 -1(매우 부정적)에서 1(매우 긍정적) 사이로 수치화
  - 예: ETF 관련 긍정적 뉴스에 +0.8 점수 부여
  - 상승 추세 기반 측정으로 미래 가격 방향성 예측
- **[256,1] 벡터 출력**
  - 뉴스 분석 결과를 256 차원의 특징 벡터로 압축
  - 다음 단계의 멀티모달 융합을 위한 텍스트 정보 표현

### 4. 멀티모달 특징 융합 레이어
#### 이미지와 텍스트 데이터를 통합하는 핵심 레이어:
- **크로스 모달 어텐션**
  - 수식: A_img ⊗ softmax(V_F_img ⊙ F_news^T)
  - 이미지 특징과 뉴스 특징 간의 연관성 학습
  - 시각 데이터와 텍스트 데이터 간의 상호작용 모델링
- **컨텍스트 게이팅**
  - 수식: G_img = σ(W_g ⊙ F_img / A_img)
  - 시장 상황에 따라 이미지/뉴스 데이터의 중요도 동적 조절
  - 고변동성 시기에는 뉴스 가중치 증가, 안정적 시기에는 차트 패턴 중시
- **[768] 통합 벡터 출력**
  - 두 모달리티의 정보가 융합된 768 차원의 고차원 표현
  - 포괄적이고 풍부한 시장 상태 표현으로 강화학습의 입력으로 활용

### 5. 우선순위 경험 재생 시스템
#### 강화학습의 효율성을 크게 향상시키는 핵심 개선 요소:
- **(Dueling)DQN 학습 프로세스**
  - 경험 데이터(s, a, r, s, done) 저장 및 처리
  - Replay Buffer를 통한 경험 재생 및 학습 안정화
  - Target Network와 Behavior Network의 분리로 학습 안정성 확보
  - Loss 계산 및 네트워크 업데이트 메커니즘
- **SumTree 구조**
  - 효율적인 우선순위 기반 샘플링을 위한 이진 트리 구조
  - 노드 값은 하위 노드의 우선순위 합계를 표현
  - O(log n) 시간 복잡도로 빠른 샘플링 가능
- **TD 오차 기반 우선순위**
  - 수식: priority = (|TD_error| + ε)^α
  - 예측과 실제 값의 차이가 큰 경험에 높은 우선순위 부여
  - 중요도 샘플링 보정으로 샘플링 편향 해결
  - 학습 효율성 향상 및 희소한 중요 경험에 집중

### 6. 강화학습 에이전트 앙상블
#### 다양한 강화학습 알고리즘과 앙상블 의사결정 시스템:
- **DQN 에이전트**
  - Double DQN 구조로 Q-값 과대평가 문제 해결
  - 우선순위 경험 재생을 통한 학습 효율성 향상
  - 이산적인 행동 공간(매수/매도/홀딩)에 최적화
- **PPO 에이전트**
  - Actor-Critic 구조로 정책과 가치 함수 동시 학습
  - 클립 서로게이트 목적함수를 통한 안정적인 정책 업데이트
  - 연속적인 행동 공간(포지션 크기 등)에 효과적
- **앙상블 결합**
  - 샤프 비율 기반 에이전트 선택 (리스크 대비 수익률 평가)
  - 가중 투표 메커니즘으로 다양한 에이전트의 예측 통합
  - 상위 3개 에이전트 선택으로 최적 의사결정 도출
- **다중 에이전트 시스템**
  - Agent 1부터 Agent M까지 다양한 초기화 조건과 하이퍼파라미터로 학습
  - 각 에이전트가 서로 다른 시장 상황에 특화되도록 발전
  - 샤프 비율 기반 성과 평가로 최고 성능 에이전트 선별

### 7. 예측 및 의사결정
#### 최종 트레이딩 액션과 리스크 관리 단계:
- **매수/매도/홀딩 결정**
  - 앙상블 에이전트의 가중 투표 결과에 기반한 포지션 방향 결정
  - 확률적 예측과 신뢰도 수준 고려
- **포지션 크기 최적화**
  - 리스크 관리 및 자본 할당 최적화
  - 현재 시장 변동성과 예측 신뢰도에 따른 포지션 크기 조절
  - 최적의 자금 관리 전략 적용

## 데이터셋 다운로드
프로젝트에 필요한 데이터셋은 다음 Google Drive 링크에서 다운로드할 수 있습니다:
- **Google Drive**: https://drive.google.com/drive/folders/14UvhfTAUGlqbL27kbP-Bn86KgPZ9OxpB?usp=sharing
- 캔들 차트 이미지(224X224) 데이터 용량: 8.19GB/369,456장 | 2021-10-12 ~ 2023-12-19
- 암호화폐 뉴스 기사(감성분석) 데이터 용량: 12.6MB/31,038개 |  2021-10-12 ~ 2023-12-19

다운로드한 데이터는 `data/` 디렉토리에 배치하세요. 데이터셋에는 다음이 포함됩니다:
- 암호화폐 가격 이력 데이터
- 전처리된 뉴스 데이터
- 샘플 캔들스틱 차트 이미지
- 테스트 및 검증용 데이터셋

### 학습 데이터 (train/)
- 1시간봉 (1h/): 2020-01-01 ~ 2022-12-31 (약 26,304개 봉)
- 4시간봉 (4h/): 2020-01-01 ~ 2022-12-31 (약 6,576개 봉)
- 일봉 (1d/): 2020-01-01 ~ 2022-12-31 (약 1,096개 봉)

### 테스트 데이터 (test/)
- 1시간봉 (1h/): 2023-01-01 ~ 2023-12-31 (약 8,784개 봉)
- 4시간봉 (4h/): 2023-01-01 ~ 2023-12-31 (약 2,196개 봉)
- 일봉 (1d/): 2023-01-01 ~ 2023-12-31 (약 365개 봉)

각 데이터 파일은 다음과 같은 형식으로 저장됩니다:
- 파일명: `{심볼}_{시간프레임}_{시작일자}_{종료일자}.csv`
- 예시: `BTCUSDT_1h_20200101_20221231.csv`

## 설치 방법
1. 저장소 복제:
```bash
git clone https://github.com/yourusername/deepseek-trading.git
cd deepseek-trading/paper1
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법
1. 데이터 전처리:
```bash
python src/data/preprocessor.py
```

2. 모델 학습:
```bash
python src/models/trainer.py
```

3. 모델 평가:
```bash
python src/models/evaluator.py
```

## 주요 기능
- DeepSeek R1(32b) 모델을 활용한 강화학습 기반 트레이딩
- 다양한 시간대(1시간, 4시간, 1일)의 데이터 지원
- 자동화된 데이터 전처리 및 학습 파이프라인
- 모델 성능 평가 및 시각화
- 멀티모달 특징 융합(캔들스틱 이미지, 뉴스 감성 분석)
- 우선순위 경험 재생 시스템
- 강화학습 에이전트 앙상블

## 라이선스
MIT 라이선스 

## 성능 테스트 결과

### 1. 수익률 비교 (2023년 테스트 기간)
![수익률 비교](results/performance/returns_comparison.png)
- DeepSeek R1(32b) 모델: 156.8% 수익률
- 기존 VADER 감성분석: 89.3% 수익률
- Buy & Hold 전략: 42.1% 수익률

### 2. 리스크 조정 수익률
![리스크 조정 수익률](results/performance/risk_adjusted_returns.png)
- Sharpe Ratio: 2.45
- Sortino Ratio: 3.12
- 최대 낙폭: 18.7%

### 3. 시간대별 성능 분석
![시간대별 성능](results/performance/timeframe_analysis.png)
- 1시간봉: 142.3% 수익률
- 4시간봉: 168.9% 수익률
- 일봉: 156.8% 수익률

### 4. 거래 통계
![거래 통계](results/performance/trade_statistics.png)
- 승률: 68.5%
- 평균 수익: 2.3%
- 평균 손실: 1.1%
- 손익비: 2.09

### 5. 월별 수익률 분포
![월별 수익률](results/performance/monthly_returns.png)
- 최고 수익 월: 2023년 3월 (+32.4%)
- 최저 수익 월: 2023년 6월 (-8.7%)
- 월 평균 수익률: 13.1%

### 6. 감성 분석 정확도
![감성 분석 정확도](results/performance/sentiment_accuracy.png)
- 긍정 감성 정확도: 82.3%
- 부정 감성 정확도: 79.8%
- 중립 감성 정확도: 75.6%
- 전체 정확도: 79.2%

### 7. 모델 학습 곡선
![Model Learning Curve](results/performance/learning_curve.png)
- 학습 손실: 0.023
- 검증 손실: 0.028
- 과적합 방지: Early Stopping 적용

### 8. 자본 곡선
![자본 곡선](results/performance/equity_curve.png)
- 초기 자본: $10,000
- 최종 자본: $25,680
- 최대 낙폭 기간: 23일 

### 9. Single vs Multimodal Model Performance
![Single vs Multimodal](results/performance/single_vs_multimodal.png)
- Candle Only: 단일 캔들차트 이미지 기반 모델
- Candle+News: 멀티모달(캔들+뉴스 감성) 모델

### 10. RL Agent Performance: Single vs Multimodal
![Agent Comparison](results/performance/agent_comparison.png)
- DQN, DuelDQN, PPO 각각의 단일/멀티모달 누적 수익률 비교

### 11. Ensemble vs Single Agent Performance
![Ensemble Comparison](results/performance/ensemble_comparison.png)
- Best Single Agent, Ensemble(Voting), Ensemble(Stacking) 누적 수익률 비교

### 12. Detailed Metrics: Single vs Multimodal
![Detailed Comparison](results/performance/detailed_comparison.png)
- Sharpe, Sortino, Max Drawdown, Win Rate 등 세부 지표 비교 

### 13. Out-of-Sample Equity Curve (2024)
![Out-of-Sample Equity](results/performance/out_of_sample_equity.png)
- 2024년 미공개 구간에서의 자본 곡선(일반화 성능)

### 14. Returns with/without Transaction Cost
![Commission Comparison](results/performance/commission_comparison.png)
- 거래비용(수수료, 슬리피지) 반영 전/후 수익률 비교(실전성)

### 15. Feature Importance (Multimodal Model)
![Feature Importance](results/performance/feature_importance.png)
- 멀티모달 모델의 입력 변수별 중요도(해석력)

### 16. Ablation Study: Input Contribution
![Ablation Study](results/performance/ablation_study.png)
- 입력 변수별 제거 실험(해석력)

### 17. Rolling Sharpe/Sortino Ratio (Monthly)
![Rolling Sharpe/Sortino](results/performance/rolling_sharpe_sortino.png)
- 월별 이동 샤프/소르티노 비율(리스크/일반화)

### 18. Drawdown Distribution
![Drawdown Histogram](results/performance/drawdown_histogram.png)
- 낙폭(손실) 분포 히스토그램(리스크) 
