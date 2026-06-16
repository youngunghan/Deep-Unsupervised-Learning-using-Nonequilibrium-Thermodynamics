# 아키텍처

> **범위:** 저장소 전체 구조, 서로 독립적인 두 구현, MNIST 백본의 설계. 수식은 [diffusion-method.md](diffusion-method.md), 이슈는 [known-issues.md](known-issues.md).
> **대상:** 코드 구조를 이해하려는 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-17.

## 1. 두 개의 독립 구현

repo에는 코드를 공유하지 않는 **두 확산 구현**이 있다.

| | MNIST 이미지 모델 | Swiss Roll 데모 |
|---|---|---|
| 진입점 | [scripts/train.py](../../scripts/train.py) | [scripts/test_swiss_roll.py](../../scripts/test_swiss_roll.py) `main()` |
| 모델 | [networks/dpm.py](../../networks/dpm.py) `DiffusionModel` | 동 파일 내부 자체 `DiffusionModel` |
| 백본 | [networks/mlp.py](../../networks/mlp.py) `MLP` | `DiffusionEncoder`(타임스텝별 head) |
| β 스케줄 | 선형 | sigmoid |
| 손실 | x0-NLL(단일 t) | 해석적 posterior의 가우시안 KL |
| 논문 충실성 | 🟠 단순화 | ✅ 더 충실 |

> 두 모델은 무관하다 — `test_swiss_roll.py`는 `networks/dpm.py`를 import하지 않는다. 손실 차이의 함의는 [diffusion-method.md](diffusion-method.md) §6.

## 2. MNIST 데이터 흐름

```
                     ┌─────────────────────────────────────────────┐
 MNIST [B,1,28,28]   │  scripts/train.py  →  scripts/trainer.py     │
   (Normalize [-1,1])│  배치마다: loss = model.cost_single_t(x)     │
        │            └─────────────────────────────────────────────┘
        ▼
 DiffusionModel.cost_single_t(x)
        │
        ├─ forward_process(x)         x⁽⁰⁾ ──► x⁽ᵗ⁾ = √ᾱ_t·x⁰ + √(1-ᾱ_t)·ε   (t 무작위, 배치당 1개)
        │
        ├─ get_mu_sigma(x⁽ᵗ⁾, t)
        │     └─ MLP(x⁽ᵗ⁾) ─► [B,2,C,n_basis,28,28] ─► temporal basis 투영 ─► (μ, σ)
        │
        └─ -log N(x⁰; μ, σ).mean()    ◄── 타깃은 깨끗한 x⁰ (🟠 known-issues D2)
```

샘플링은 역방향으로 `get_mu_sigma`를 반복한다([diffusion-method.md](diffusion-method.md) §4).

## 3. MNIST 백본 (MLP)

[networks/mlp.py](../../networks/mlp.py) `MLP.forward()`는 두 경로를 합산 후 투영한다.

```
 x⁽ᵗ⁾ ─┬─► [MSC 경로]  num_layers × MultiscaleConvolution ──┐
        │                                                    (+)──► final_conv ──► [B, 2·C·n_basis, 28,28]
        └─► [Dense 경로] num_layers × (Conv2d 1×1 + Tanh) ───┘
```

- `MultiscaleConvolution.forward()`: 각 scale에서 `avg_pool(2^scale)` → `conv5×5` → `softplus(x-1)` → bilinear 업샘플 → 합산 후 `/num_scales`.
- 출력 `2·n_colors·n_temporal_basis` 채널 = (μ 계수, log σ 계수) × `n_temporal_basis`.
- 설계 메모: 🟠 `Tanh`가 계수를 $[-1,1]$로 제한해 표현력을 줄인다; 🟠 200층 × 두 경로라 메모리가 커서 `batch_size`가 8로 강제된다(배치 상한은 [known-issues.md](known-issues.md) §2 D7, 작은 이미지 제약은 §3 D9).

## 4. Temporal 가우시안 basis

[networks/dpm.py](../../networks/dpm.py) `generate_temporal_basis()`가 로그-간격 중심·적응 폭의 가우시안 뱅크(합=1 정규화)를 버퍼 `temporal_basis`(`[n_basis, T]`)로 등록한다. `get_mu_sigma()`는 현재 $t$에서 basis를 축약해 MLP 계수를 시간 가중합 → 하나의 네트워크가 $t$에 따라 부드럽게 변하는 μ/σ를 출력한다(논문 Appendix D.2.1 아이디어).

## 5. 모듈 지도

| 모듈 | 책임 |
|---|---|
| [networks/dpm.py](../../networks/dpm.py) | 확산 모델·순/역방향·손실·샘플러 |
| [networks/mlp.py](../../networks/mlp.py) | MLP·MultiscaleConvolution 백본 |
| [scripts/train.py](../../scripts/train.py) | 학습 진입점·데이터·옵티마이저·재개 |
| [scripts/trainer.py](../../scripts/trainer.py) | 학습 루프·검증·체크포인트·TensorBoard |
| [scripts/validate.py](../../scripts/validate.py) | 평균 검증 손실 |
| [scripts/infer.py](../../scripts/infer.py) | inpaint/denoise 유틸(no `__main__`) |
| [scripts/test_swiss_roll.py](../../scripts/test_swiss_roll.py) | 독립 2D 데모(모델·지표·시각화·`__main__`) |
| [options/train_options.py](../../options/train_options.py) | argparse 설정 |
| [dataset/dataloader.py](../../dataset/dataloader.py) | 라벨 제거 래퍼 |
| [utils.py](../../utils.py) | 로거·별도 샘플러 |

## 관련 문서

- [diffusion-method.md](diffusion-method.md) · [known-issues.md](known-issues.md) · [reference/api-reference.md](../reference/api-reference.md)
