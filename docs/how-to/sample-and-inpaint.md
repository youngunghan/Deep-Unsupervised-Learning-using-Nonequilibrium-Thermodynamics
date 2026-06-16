# 샘플 생성 & inpainting

> **범위:** 학습된 MNIST 모델로 샘플을 생성하고, [scripts/infer.py](../../scripts/infer.py)의 inpainting/denoising 유틸을 사용하는 방법. 역방향 과정의 이론은 [explanation/diffusion-method.md](../explanation/diffusion-method.md) §4.
> **대상:** 학습 후 생성/조건부 생성을 시도하는 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-17.

## 1. 두 가지 샘플러 (중요)

🟠 repo에는 동작이 다른 **두 샘플러**가 있다:

| 함수 | 위치 | 루프 범위 | 결과 |
|---|---|---|---|
| `DiffusionModel.sample()` | [networks/dpm.py](../../networks/dpm.py) | `t = T-1 … min_t` | `t=min_t`에서 멈춤 → 잔여 잡음 |
| `generate_samples()` | [utils.py](../../utils.py) | `t = T-1 … 0` | `t=0`까지 디노이즈 |

학습 미리보기([scripts/trainer.py](../../scripts/trainer.py) `train_epoch()`)는 전자를 쓴다. 같은 모델이라도 두 함수의 출력 궤적이 다르다 — 자세한 배경은 [explanation/known-issues.md](../explanation/known-issues.md) §1 D1.

## 2. 체크포인트 로드 후 샘플링

`infer.py`에는 `__main__`이 없으므로(라이브러리 유틸), 다음처럼 직접 호출한다:

```python
import torch
from networks.dpm import DiffusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffusionModel(spatial_width=28, n_colors=1, n_temporal_basis=10,
                       trajectory_length=1000, device=device).to(device)
ckpt = torch.load("checkpoints/diffusion_default_best.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with torch.no_grad():
    samples = model.sample(16)        # [16, 1, 28, 28]
```

> 모델 생성 인자는 학습 때와 **동일**해야 state_dict가 맞는다([reference/api-reference.md](../reference/api-reference.md) §1).

## 3. Inpainting / denoising 유틸

[scripts/infer.py](../../scripts/infer.py)는 가이드 역방향 스텝을 위한 함수를 제공한다:

- `diffusion_step(Xmid, t, get_mu_sigma, denoise_sigma, mask, XT, device)` — 한 역방향 스텝. `mask`가 주어지면 마스크 영역을 `XT`로 고정하고, `denoise_sigma`가 주어지면 정밀도 가중 평균으로 디노이징한다.
- `generate_inpaint_mask(n_samples, n_colors, spatial_width)` — 이미지 오른쪽 절반을 마스킹한 평탄 boolean 배열을 만든다.

```python
import torch
from scripts.infer import diffusion_step, generate_inpaint_mask

mask = generate_inpaint_mask(16, 1, 28)        # 평탄 bool, 오른쪽 절반 True
XT = real_images                                # [16,1,28,28] 보존할 원본
Xmid = torch.randn(16, 1, 28, 28, device=device)
for t in reversed(range(model.min_t, model.trajectory_length)):
    tt = torch.tensor([t], device=device)
    Xmid = diffusion_step(Xmid, tt, model.get_mu_sigma,
                          denoise_sigma=None, mask=mask, XT=XT, device=device)
```

> ✅ **2026-06-16 수정 반영 [검증 반영 #1] (2026-06-16).** 이 경로에는 두 버그가 있었다: ① `generate_inpaint_mask`의 `spatial_width/2`(Python 3 float 인덱스 → `TypeError`) → `//`로 수정; ② `diffusion_step`의 `mu.flat[mask]`(torch 텐서엔 `.flat` 없음 → `AttributeError`) → 마스크 reshape + `torch.where`로 수정. CPU smoke test로 동작 확인. 상세 [explanation/known-issues.md](../explanation/known-issues.md) §1 F1·F2.

## 4. 결과 시각화

`infer.py` `plot_images(images, title, ncols)`로 격자 출력하거나, 학습 중이라면 TensorBoard `samples/generated` 패널을 본다.

> 🟢 두 샘플러 모두 출력을 픽셀 범위로 clamp/정규화하지 않는다(값이 $[-1,1]$ 밖일 수 있음). 학습 중 미리보기는 `make_grid(..., normalize=True)`(`scripts/trainer.py`)로 정규화되지만, 수동 저장/표시 시에는 직접 clamp 또는 normalize가 필요하다.

## 관련 문서

- [how-to/train-mnist.md](train-mnist.md) · [explanation/diffusion-method.md](../explanation/diffusion-method.md) · [explanation/known-issues.md](../explanation/known-issues.md)
