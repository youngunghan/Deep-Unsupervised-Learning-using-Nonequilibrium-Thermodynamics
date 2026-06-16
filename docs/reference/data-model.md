# 데이터 모델 레퍼런스

> **범위:** 데이터셋, 전처리, 텐서 형상, 노이즈 스케줄, 체크포인트 구조. 파이프라인 설명은 [explanation/architecture.md](../explanation/architecture.md).
> **대상:** 데이터/텐서 흐름을 확인하는 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-17.

## 1. 데이터셋

| 데이터셋 | 출처 | 형상 | 전처리 |
|---|---|---|---|
| MNIST | `torchvision.datasets.MNIST`(자동 다운로드) | `[1,28,28]` | `ToTensor` → `Normalize((0.5,),(0.5,))` = $[-1,1]$ |
| Swiss Roll | `sklearn.datasets.make_swiss_roll`(즉석 생성) | `[N,2]` | 차원 `[2,0]` 선택 후 `/10·[1,-1]` ≈ $[-2,2]$ |

- MNIST 라벨은 [dataset/dataloader.py](../../dataset/dataloader.py) `CustomDataset`이 제거한다(무조건부 생성).
- Swiss Roll은 [scripts/test_swiss_roll.py](../../scripts/test_swiss_roll.py) `create_swiss_roll_dataset()`이 매 스텝 새로 생성한다.

## 2. 노이즈 스케줄

### 2.1 MNIST ([networks/dpm.py](../../networks/dpm.py) `_initialize_diffusion_params()`)

| 버퍼 | 정의 | 형상 |
|---|---|---|
| `beta` | `linspace(beta_start, beta_end, T)` (선형) | `[T]` |
| `alpha` | `1 - beta` | `[T]` |
| `alpha_cum` | `cumprod(alpha)` = $\bar\alpha_t$ | `[T]` |

기본 $\beta\in[0.01,0.05]$, $T=1000$(학습 스크립트 기본값. `DiffusionModel` 클래스 시그니처 자체의 기본 인자는 `trajectory_length=2000`이지만 `scripts/train.py`가 항상 `args.trajectory_length`로 덮어써 실제 MNIST 실행 T는 1000이다 — [explanation/known-issues.md](../explanation/known-issues.md) §3 D8). 평균 $\beta\approx0.03$ → $\bar\alpha_T\approx e^{-30}\approx0$ 이라 $x^{(T)}\approx\mathcal N(0,I)$로 prior와 정합한다. 🟠 DDPM의 흔한 $[10^{-4},0.02]$보다 공격적이고, 2015 논문은 스케줄을 *학습*한다([explanation/known-issues.md](../explanation/known-issues.md) §2 D6).

### 2.2 Swiss Roll ([scripts/test_swiss_roll.py](../../scripts/test_swiss_roll.py) `DiffusionModel`)

`beta = sigmoid(linspace(-18,10,T)) · (beta_max - beta_min) + beta_min`, $\beta_\min=10^{-5}$, $\beta_\max=0.3$, $T=40$. (sigmoid 스케줄)

## 3. 텐서 형상 (MNIST forward)

```
x⁽⁰⁾            [B, 1, 28, 28]      입력(정규화 [-1,1])
forward_process → noisy_x [B,1,28,28], noise [B,1,28,28], t [1], alpha_cum_t [1,1,1,1]
MLP(x_t)        [B, 20, 28, 28]     20 = 2·n_colors·n_temporal_basis
 reshape        [B, 2, 1, 10, 28, 28]   (μ계수, logσ계수) × n_basis
get_mu_sigma →  mu [B,1,28,28], sigma [B,1,28,28]
```

> 🟠 `t`의 형상이 `[1]`이라 배치 전체가 같은 타임스텝을 공유한다([explanation/known-issues.md](../explanation/known-issues.md) §1 D3).

## 4. 체크포인트 구조

[scripts/trainer.py](../../scripts/trainer.py) `train_epoch()`가 저장하는 dict:

| 키 | 내용 |
|---|---|
| `epoch`, `batch` | 재개 위치 |
| `model_state_dict` | 모델 가중치 |
| `optimizer_state_dict` | Adam 상태 |
| `val_loss` | 검증 손실 |
| `args` | 학습 인자 네임스페이스 |

Swiss Roll([scripts/test_swiss_roll.py](../../scripts/test_swiss_roll.py) `save_model_checkpoint()`)은 `epoch`·`model_state_dict`·`optimizer_state_dict`·`metrics`를 저장한다.

## 관련 문서

- [explanation/architecture.md](../explanation/architecture.md) · [explanation/diffusion-method.md](../explanation/diffusion-method.md)
