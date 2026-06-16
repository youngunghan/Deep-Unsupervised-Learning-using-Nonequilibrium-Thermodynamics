# Deep Unsupervised Learning using Nonequilibrium Thermodynamics — 개발자 문서

Sohl-Dickstein et al. 2015([arXiv:1503.03585](https://arxiv.org/abs/1503.03585))의 PyTorch 재구현 — DDPM의 전신인 비평형 열역학 확산모델. 상위: [../README.md](../README.md).

## 문서 목록 (Diátaxis)

### Tutorials — 처음 따라하기

| 문서 | 설명 |
|---|---|
| [tutorials/quickstart.md](tutorials/quickstart.md) | 설치 → MNIST 학습 → TensorBoard로 샘플 확인 (happy path) |

### How-to — 과업 가이드

| 문서 | 설명 |
|---|---|
| [how-to/train-mnist.md](how-to/train-mnist.md) | MNIST 학습 실행·체크포인트 재개·하이퍼파라미터 변경 |
| [how-to/sample-and-inpaint.md](how-to/sample-and-inpaint.md) | 학습된 모델로 샘플 생성, inpainting/denoising 유틸 사용 |
| [how-to/run-swiss-roll.md](how-to/run-swiss-roll.md) | 2D Swiss Roll 데모 실행·평가지표·시각화 |

### Reference — 조회용 명세

| 문서 | 설명 |
|---|---|
| [reference/configuration.md](reference/configuration.md) | CLI 인자·`environment.yml`·셸 스크립트 설정 (표) |
| [reference/api-reference.md](reference/api-reference.md) | `DiffusionModel`·`MLP`·`infer` 함수 명세 (파라미터 표) |
| [reference/data-model.md](reference/data-model.md) | 데이터셋·텐서 형상·노이즈 스케줄 (표) |

### Explanation — 깊은 설명

| 문서 | 설명 |
|---|---|
| [explanation/architecture.md](explanation/architecture.md) | 전체 구조: 두 구현 대비·모델 백본·데이터 흐름 |
| [explanation/diffusion-method.md](explanation/diffusion-method.md) | 확산모델·비평형 열역학·순방향/역방향/변분 목적함수 (수학) |
| [explanation/known-issues.md](explanation/known-issues.md) | 알려진 이슈·수정 내역·재현성 체크리스트 |
