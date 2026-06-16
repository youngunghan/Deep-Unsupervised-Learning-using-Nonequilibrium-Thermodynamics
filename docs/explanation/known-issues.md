# 알려진 이슈 · 수정 내역 · 재현성

> **범위:** 2026-06-16/17 코드 리뷰 결과. 수정된 버그(상태 ✅)와 문서화만 한 항목(🔴/🟠/🟢)을 구분한다. 방법론 배경은 [diffusion-method.md](diffusion-method.md).
> **대상:** 코드를 신뢰성 있게 쓰거나 개선하려는 개발자.
> **상태:** 리뷰 반영 — 기준일 2026-06-17. 모든 항목은 코드 정독 + CPU smoke test로 확인(추측 0).

## 1. 정확성 · 충실성

### 1.1 수정 완료 (repo에 반영)

| # | 상태 | 이슈 | 위치 | 수정 |
|---|---|---|---|---|
| F1 | ✅ `[검증 반영 #1]` (2026-06-16) | Python-2식 `spatial_width/2` float 슬라이스 인덱스 → Py3 `TypeError` | [scripts/infer.py](../../scripts/infer.py) `generate_inpaint_mask()` | `//` 정수 나눗셈 |
| F2 | ✅ `[검증 반영 #1]` (2026-06-16) | torch 텐서엔 `.flat` 없음(`mu.flat[mask]`) → `AttributeError` | [scripts/infer.py](../../scripts/infer.py) `diffusion_step()` | 마스크 reshape + `torch.where` |
| F3 | ✅ (2026-06-16) | README가 `python scripts/infer.py`를 추론 실행으로 안내(파일에 `__main__` 없음) | [../README.md](../../README.md) | 실제 샘플링 경로 설명으로 정정 |
| F4 | ✅ (2026-06-16) | `--val_interval` help "epochs"인데 실제 배치 단위 | [options/train_options.py](../../options/train_options.py) | help "batches"로 정정 |
| F5 | ✅ (2026-06-16) | README "Swoss Roll" 오타·"MNIST 전처리" 오해 | [../README.md](../../README.md) | 오타·안내 정정 + "PyTorch 재구현" 명시 |
| F6 | ✅ (2026-06-17) | `train.sh`/CLI가 `--hidden_channels`, `--num_layers`, `--beta_start`, `--beta_end`, `--min_t`를 넘기지만 `scripts/train.py`가 `DiffusionModel` 생성자에 전달하지 않음 | [scripts/train.py](../../scripts/train.py) | 모델/확산 인자를 모두 전달하고 CPU fallback 시 `args.device`도 동기화 |
| F7 | ✅ (2026-06-17) | `MultiscaleConvolution`이 생성 시점의 CUDA 추측 device를 포착해 `model.to(cpu)` 이후에도 입력을 CUDA로 이동 → CPU weight와 충돌 | [networks/mlp.py](../../networks/mlp.py) | `forward()`에서 실제 모듈 파라미터 device를 따르고, `MLP` 생성 시 명시 device를 전체 하위 모듈에 적용 |

> 수정 후 `py_compile` + CPU smoke test(`forward_process`/`get_mu_sigma`/`cost_single_t`/`sample` + 수정된 inpaint/denoise)로 무오류 확인.

### 1.2 문서화만 (동작 변경 — 재학습/검증 필요)

- 🔴 **D1 — 절단된 샘플러 / 죽은 분기.** `DiffusionModel.sample()`이 `t=min_t`(기본 100)에서 멈춰 `t=0`에 도달하지 않는다 → 출력에 잔여 잡음, `else: x=μ`는 도달 불가. `generate_samples()`(utils.py)는 `t=0`까지 돈다 → **두 샘플러 불일치**. `min_t`가 학습도 게이팅하므로 단순히 낮추면 미학습 영역을 샘플링하게 된다. *수정 방향*: 둘 다 `t=0`까지, `min_t` 의미 재정의. 근거: 논문 Appendix(역방향 $t=T..1$), DDPM Algorithm 2.
- 🔴 **D2 — 목적함수↔샘플러 불일치.** `cost_single_t()`는 $\mu$를 $x^{(0)}$-예측기로 학습하나 `sample()`은 $\mu$를 역방향 평균으로 쓴다. *수정 방향*: $x^{(0)}$-추정을 posterior 평균으로 변환하거나 DDPM ε-예측 + 스텝별 KL/MSE로 전환. 근거: 논문 $q(x^{(t-1)}|x^{(t)},x^{(0)})$, DDPM Eq. 11–12. [diffusion-method.md](diffusion-method.md) §5.
- 🟠 **D3 — 미니배치당 타임스텝 하나.** `forward_process()`가 `t`를 `(1,)`로 뽑아 배치 전체가 공유 → $\mathbb{E}_t$가 1-샘플 추정. *수정 방향*: 예제마다 `t` 샘플링 후 `get_t_weights()`·`σ` clamp을 배치 `t`로 일반화. 근거: DDPM은 예제별 `t`.

## 2. 재현성 · 의존성

- 🟠 **D4 — 시드 부재.** [scripts/train.py](../../scripts/train.py) `main()`이 시드를 설정하지 않아 비결정적. *수정*: `torch`/`numpy`/`cuda` 시드 + `--seed`.
- 🟠 **D5 — 의존성 미핀 / `scipy` 미선언.** [environment.yml](../../environment.yml)이 `python`·`cudatoolkit`만 핀; `scipy`(Swiss Roll 사용)는 미선언. *수정*: 정확한 버전 핀 + `scipy` 추가.
- 🟠 **D6 — 공격적/고정 β 스케줄.** $[0.01,0.05]$는 DDPM $[10^{-4},0.02]$보다 거칠고, `train.sh` 주석 "from the paper"는 오해(2015 논문은 스케줄을 *학습*). *수정*: 주석 정정·스케줄 문서화/튜닝.
- 🟠 **D7 — 하드코딩된 환경.** `train.sh`/`run_swiss_roll.sh`의 `CUDA_VISIBLE_DEVICES=2`, `train.py`의 `batch_size≤8` 캡, `train.sh` 절대경로 주석 잔재. *수정*: GPU id·배치 캡 설정화.

## 3. 경미 · 품질

- 🟢 **D8 — 기본값 불일치.** `DiffusionModel` 기본 `trajectory_length=2000` vs 실사용 `1000`(항상 덮어씀). *수정*: 1000으로 통일.
- 🟢 **D9 — 작은 이미지 제약.** `num_scales=4` + reflect 패딩이면 가장 깊은 scale이 `floor(spatial_width/8)`이고 padding 2가 필요하므로 `spatial_width >= 24`가 안전하다(8×8/16×16은 너무 작아 `RuntimeError`). *수정*: `num_scales`를 입력 크기로 제한.
- 🟢 **D11 — Swiss Roll 디바이스.** 자체 `DiffusionModel`이 `beta/alpha`를 일반 CPU 텐서로 두고 인코더만 이동 → GPU에서 불일치 가능(미검증). *수정*: 버퍼 등록/모델 이동.

## 4. 재현성 체크리스트

| 영역 | 현재 | 액션 |
|---|---|---|
| 결정성 | 🟠 시드 없음 | 시드 + `--seed`(D4) |
| 의존성 핀 | 🟠 대부분 미핀 | 버전 핀 + `scipy`(D5) |
| 하드웨어 | 🟠 `cuda` 기본·GPU id 고정 | device/GPU/배치 설정화(D7) |
| 사전학습 가중치 | 🔴 없음 | 체크포인트 + config 공개 |
| 학습 결과 | 🔴 곡선/샘플 미공개 | MNIST 샘플·Swiss Roll 지표 보고 |
| 샘플러 | 🔴 두 샘플러·절단 | 하나로 통일·`t=0`까지(D1) |
| 원본 코드 | 🟢 Theano/Blocks/Fuel+Py2(미유지) | 이 PyTorch 재작성 사용(결과 비동일) |

**현재 코드 실행 최소 절차**: `conda env create -f environment.yml` → `./train.sh` 또는 `./run_swiss_roll.sh`(필요 시 `--device cpu`·`CUDA_VISIBLE_DEVICES` 덮어쓰기). **로컬 검증(2026-06-17)**: 안전 수정 후 PyTorch 2.9.1 / NumPy 2.2.6에서 import·컴파일 + CPU smoke 통과; 전체 학습 수렴·MNIST 샘플 품질은 미검증.

## 5. 참고 자료

- Sohl-Dickstein et al. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics.* ICML — [arXiv:1503.03585](https://arxiv.org/abs/1503.03585). 원본 코드: `github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models`(Theano/Blocks/Fuel, Py2).
- Ho, Jain, Abbeel (2020). *Denoising Diffusion Probabilistic Models.* — [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).
- `lucidrains/denoising-diffusion-pytorch` — 고-star 참조 구현.
- Theano 폐기/후속: [PyMC 이력](https://www.pymc.io/about/history.html) · [PyTensor](https://github.com/pymc-devs/pytensor).

## 관련 문서

- [diffusion-method.md](diffusion-method.md) · [architecture.md](architecture.md) · [reference/configuration.md](../reference/configuration.md)
