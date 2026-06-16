# API 레퍼런스

> **범위:** 주요 클래스·함수의 시그니처와 반환값. 동작·수식 설명은 [explanation/diffusion-method.md](../explanation/diffusion-method.md), 사용 예는 [how-to/](../how-to/).
> **대상:** 코드를 직접 호출/확장하는 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-17.

## 1. DiffusionModel ([networks/dpm.py](../../networks/dpm.py))

MNIST 이미지 확산모델(`nn.Module`).

### 1.1 생성자

| 파라미터 | 기본 | 설명 |
|---|---|---|
| `spatial_width` | — | 입력 해상도 |
| `n_colors` | — | 채널 수 |
| `n_temporal_basis` | `10` | temporal basis 개수 |
| `trajectory_length` | `2000` | 확산 스텝 $T$(실행 시 보통 `1000`) |
| `beta_start` / `beta_end` | `0.01` / `0.05` | 선형 β 스케줄 |
| `mlp_layers` | `200` | 백본 층 수 |
| `mlp_hidden_channels` | `128` | 백본 은닉 채널 |
| `min_t` | `100` | 학습/샘플링 타임스텝 하한 |
| `eps` | `1e-5` | 수치 안정 상수 |
| `device` | `None` | 미지정 시 cuda-if-available |

### 1.2 메서드

| 메서드 | 반환 | 설명 |
|---|---|---|
| `forward_process(x, t=None)` | `(noisy_x, noise, t, alpha_cum_t)` | 닫힌 형식 순방향 잡음화. `t=None`이면 `[min_t,T)`에서 무작위(배치당 하나) |
| `get_mu_sigma(x_t, t)` | `(mu, sigma)` | 역방향 파라미터 예측. temporal basis 투영 + 스케일/clamp |
| `cost_single_t(x)` | scalar tensor | 학습 손실 = 깨끗한 `x`에 대한 음의 로그우도 |
| `sample(batch_size)` | tensor `[B,C,W,W]` | 역방향 샘플링. 🟠 `t=min_t`에서 멈춤 |
| `generate_temporal_basis(T, n_basis)` | tensor `[n_basis,T]` | 정규화된 가우시안 basis |
| `get_t_weights(t)` | tensor `[T,1]` | 단일 `t`의 one-hot(배치 `t` 미지원) |

> 🟠 `cost_single_t`의 손실 타깃은 $x^{(0)}$(깨끗한 입력)이라 $\mu$가 $x^{(0)}$-예측기로 학습된다. 샘플러는 같은 $\mu$를 역방향 평균으로 쓴다(불일치). [explanation/known-issues.md](../explanation/known-issues.md) §1 D2.

## 2. 백본 ([networks/mlp.py](../../networks/mlp.py))

### 2.1 MLP

| 파라미터 | 기본 | 설명 |
|---|---|---|
| `num_channels` | `1` | 입력 채널 |
| `num_layers` | `200` | 각 경로 층 수 |
| `num_output_channels` | `20` | `2·n_colors·n_temporal_basis` |
| `hidden_channels` | `128` | 은닉 채널 |
| `activation` | `nn.Tanh` | 🟠 출력 제한 |
| `reduction_factor` | `2` | 최종층 채널 축소 |
| `device` | `None` | 지정 시 MLP 전체 하위 모듈을 해당 device로 이동 |

`forward(x)` → `final_conv(x_msc + x_dense)`: 멀티스케일 경로와 dense(1×1) 경로를 합산 후 투영.

### 2.2 MultiscaleConvolution

| 파라미터 | 기본 | 설명 |
|---|---|---|
| `num_scales` | `4` | 스케일 수(2^scale 다운샘플) |
| `filter_size` | `5` | 커널 |
| `padding_mode` | `'reflect'` | 🟠 작은 입력에서 제약(아래) |
| `activation` | shifted softplus | `softplus(x-1)` |

> 🟠 `num_scales=4` + reflect 패딩이면 `spatial_width >= 24`가 안전하다(가장 깊은 scale이 `floor(W/8)`이고 reflect padding 2가 필요). [explanation/known-issues.md](../explanation/known-issues.md) §3 D9.

## 3. 추론 유틸 ([scripts/infer.py](../../scripts/infer.py))

| 함수 | 시그니처 | 설명 |
|---|---|---|
| `diffusion_step` | `(Xmid, t, get_mu_sigma, denoise_sigma, mask, XT, device)` | 한 역방향 스텝(옵션 denoise/inpaint) |
| `generate_inpaint_mask` | `(n_samples, n_colors, spatial_width)` | 오른쪽 절반 마스크(평탄 bool) |
| `plot_images` | `(images, title=None, ncols=6)` | matplotlib 격자 출력 |

> ✅ `diffusion_step`/`generate_inpaint_mask`는 2026-06-16에 torch `.flat`·Python 3 `//` 버그를 수정함 [검증 반영 #1] (2026-06-16).

## 4. 유틸 / 학습 ([utils.py](../../utils.py), [scripts/](../../scripts/))

| 함수 | 위치 | 설명 |
|---|---|---|
| `generate_samples(model, n_samples=16)` | [utils.py](../../utils.py) | 🟠 `t=0`까지 도는 별도 샘플러 |
| `setup_logger(exp_name)` | [utils.py](../../utils.py) | 파일+콘솔 로거 |
| `train_epoch(...)` | [scripts/trainer.py](../../scripts/trainer.py) | 학습 루프 |
| `validate(model, val_loader, device)` | [scripts/validate.py](../../scripts/validate.py) | 평균 검증 손실 |
| `CustomDataset(dataset)` | [dataset/dataloader.py](../../dataset/dataloader.py) | 라벨 제거 래퍼 |

## 관련 문서

- [explanation/architecture.md](../explanation/architecture.md) · [explanation/diffusion-method.md](../explanation/diffusion-method.md)
