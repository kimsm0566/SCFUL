# 외부 baseline 코드

이 디렉터리는 연구 비교를 위해 가져온 외부 저장소를 보관한다.

- `SCU/`: semantic communication unlearning baseline clone
- `FedOSD/`: federated unlearning baseline clone

새 연구 구현은 `src/`에만 작성하며, 이 디렉터리의 코드를 import하지 않는다.
각 baseline을 실행하거나 원본 구현을 확인할 때는 해당 clone 내부의 README와 entry
point를 사용한다. 두 clone의 Git history와 기존 working-tree 변경 상태는 provenance를
위해 보존한다.

## GitHub 배포

외부 저장소 전체를 복사하지 않고 다음 원본 커밋을 submodule로 고정한다.

| 경로 | 원본 저장소 | 고정 커밋 |
| --- | --- | --- |
| `SCU/` | https://github.com/wwq5-code/SCU | `3a5d7ae6d7ef2e29ef3f560dbee01d4ca45ad4ce` |
| `FedOSD/` | https://github.com/zibinpan/FedOSD | `41cc10635c6f2396795d0d9c6239ea181b7c9ee6` |

SCU working tree에 이미 있던 torchvision `make_grid` 호환성 수정과 회귀 테스트는
`patches/scu-torchvision-compatibility.patch`로 보존한다. 새 clone에서 이 수정이
필요하면 저장소 최상위에서 다음을 실행한다. 이미 수정된 기존 working tree에는
중복 적용하지 않는다.

```bash
git -C third_party/SCU apply --check ../patches/scu-torchvision-compatibility.patch
git -C third_party/SCU apply ../patches/scu-torchvision-compatibility.patch
docker compose exec -T scful-dev python -m unittest discover -s third_party/SCU/tests
```

각 외부 baseline의 이용 조건은 원본 저장소를 따른다. 외부 clone 안의 다운로드
데이터와 캐시는 배포하지 않는다.
