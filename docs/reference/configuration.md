# 설정 레퍼런스

> **범위:** CLI 인자([options/train_options.py](../../options/train_options.py)·[scripts/test_swiss_roll.py](../../scripts/test_swiss_roll.py)), conda 환경, 셸 런처 설정. 사용 방법은 [how-to/train-mnist.md](../how-to/train-mnist.md).
> **대상:** 학습을 설정·튜닝하는 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-17.

## 1. CLI 인자 (train_options)

[options/train_options.py](../../options/train_options.py) `train_options()` — MNIST 학습용.

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--batch_size` | int | `32` | 배치 크기. 🟠 `train.py`가 8로 상한 |
| `--learning_rate` | float | `1e-5` | Adam 학습률 |
| `--epochs` | int | `5` | 에폭 수 |
| `--spatial_width` | int | `28` | 입력 해상도(정사각) |
| `--n_colors` | int | `1` | 채널 수 |
| `--n_temporal_basis` | int | `10` | temporal 가우시안 basis 개수 |
| `--trajectory_length` | int | `1000` | 확산 스텝 수 $T$ |
| `--hidden_channels` | int | `128` | 백본 은닉 채널 |
| `--num_layers` | int | `200` | 두 경로 각각의 층 수 |
| `--beta_start` | float | `0.01` | β 스케줄 시작 |
| `--beta_end` | float | `0.05` | β 스케줄 끝 |
| `--min_t` | int | `100` | 학습/샘플링 타임스텝 하한 |
| `--device` | str | `cuda` | `cuda` 또는 `cpu` |
| `--exp_name` | str | `diffusion_default` | 실험명(파일 접두) |
| `--val_interval` | int | `10` | 검증 주기 — **배치 단위** ✅(2026-06-16 help 문구 정정) |
| `--save_dir` | str | `checkpoints` | 체크포인트 디렉터리 |
| `--continue_train` | flag | `False` | 체크포인트 재개 |
| `--checkpoint_path` | str | — | 재개할 체크포인트 경로 |

> ✅ 2026-06-17 재검토에서 `--hidden_channels`, `--num_layers`, `--beta_start`, `--beta_end`, `--min_t`가
> 실제 `DiffusionModel` 생성자에 전달되도록 정합화했다. 단 `--batch_size`는 여전히 `scripts/train.py`가 8로 상한한다.

> 🟠 `DiffusionModel` 생성자 기본 `trajectory_length`는 `2000`이지만 모든 실행 경로가 인자로 `1000`을 넘긴다([explanation/known-issues.md](../explanation/known-issues.md) §3 D8).

## 2. Swiss Roll 인자 (test_swiss_roll)

[scripts/test_swiss_roll.py](../../scripts/test_swiss_roll.py) `parse_training_args()`.

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--num_timesteps` | `40` | 확산 스텝 $T$ |
| `--hidden_dim` | `256` | 은닉 차원 |
| `--data_dim` | `2` | 데이터 차원(2D) |
| `--beta_min` / `--beta_max` | `1e-5` / `3e-1` | sigmoid β 스케줄 범위 |
| `--beta_schedule_min` / `--beta_schedule_max` | `-18` / `10` | sigmoid 입력 범위 |
| `--learning_rate` | `2e-4` | Adam 학습률 |
| `--batch_size` | `128000` | 매 스텝 생성하는 점 수 |
| `--num_epochs` | `300000` | 학습 스텝 수 |
| `--eval_interval` | `3000` | 평가/시각화 주기 |

## 3. conda 환경 (environment.yml)

[environment.yml](../../environment.yml) — 환경명 `deep_thermo`.

| 패키지 | 핀 | 비고 |
|---|---|---|
| `python` | `=3.9` | |
| `pytorch`, `torchvision` | 미핀 | 🟠 버전 미고정 |
| `cudatoolkit` | `=11.8` | GPU 런타임 |
| `numpy`, `matplotlib`, `scikit-learn`, `pandas` | 미핀 | |
| `tqdm`, `jupyter`, `ipykernel` | 미핀 | |
| `tensorboard`(pip) | 미핀 | |
| `scipy` | **미선언** | 🟠 Swiss Roll에서 사용 — [explanation/known-issues.md](../explanation/known-issues.md) §2 D5 |

## 4. 셸 런처 & 환경변수

| 위치 | 설정 | 비고 |
|---|---|---|
| [train.sh](../../train.sh) | `CUDA_VISIBLE_DEVICES=2` | 🟠 GPU 인덱스 하드코딩 |
| [run_swiss_roll.sh](../../run_swiss_roll.sh) | `CUDA_VISIBLE_DEVICES=2` | 🟠 동일 |
| [scripts/train.py](../../scripts/train.py) `main()` | `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` | CUDA 단편화 완화 |
| [scripts/train.py](../../scripts/train.py) `main()` | `cudnn.benchmark=True` | |

## 관련 문서

- [how-to/train-mnist.md](../how-to/train-mnist.md) · [reference/api-reference.md](api-reference.md) · [explanation/known-issues.md](../explanation/known-issues.md)
