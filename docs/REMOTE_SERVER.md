# 원격 서버 실행 가이드

로컬 PC에서 LLM 추론이 부담스러울 때, GPU가 있는 원격 서버에서 이 프로젝트를 실행하는 절차입니다.
가장 안전한 기본 방식은 **Streamlit을 서버 내부 주소(`127.0.0.1`)에만 띄우고 SSH 터널로 접속**하는 것입니다.

## 0. 권장 서버 사양

- OS: Ubuntu 22.04 LTS 또는 24.04 LTS
- Python: 3.11 이상
- RAM: 최소 16GB, 권장 32GB 이상
- GPU: `qwen3:8b`는 8GB VRAM급에서도 가능, 14B/32B 모델은 L4/A100급 권장
- 필수 도구: `git`, `curl`, `uv`
- LLM 백엔드: `ollama` 또는 `vllm`

## 1. 서버 접속

로컬 터미널에서 서버에 접속합니다.

```bash
ssh <USER>@<SERVER_IP>
```

SSH 키를 별도 지정해야 한다면 다음처럼 접속합니다.

```bash
ssh -i ~/.ssh/<KEY_FILE> <USER>@<SERVER_IP>
```

## 2. 기본 패키지 설치

Ubuntu 기준입니다.

```bash
sudo apt update
sudo apt install -y git curl build-essential tmux
```

`uv`를 설치합니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

Python 3.11이 없다면 `uv`로 설치합니다.

```bash
uv python install 3.11
```

## 3. 프로젝트 내려받기

원격 저장소 URL을 알고 있다면 서버에서 clone 합니다.

```bash
git clone <REPOSITORY_URL>
cd 8TH-KB-AI-CHALLENGE
```

이 문서가 들어간 브랜치를 사용할 경우:

```bash
git fetch origin codex/remote-server-runbook
git switch codex/remote-server-runbook
```

아직 브랜치를 원격 저장소에 올리지 않았다면, 로컬 PC에서 먼저 실행합니다.

```bash
git push -u origin codex/remote-server-runbook
```

## 4. Python 의존성 설치

Streamlit UI까지 실행하려면:

```bash
uv sync --extra ui
```

HuggingFace persona 데이터를 서버에서 직접 내려받으려면:

```bash
uv sync --extra ui --extra personas
```

dense retrieval 실험까지 필요하면:

```bash
uv sync --extra ui --extra personas --extra dense
```

## 5. 환경변수 설정

```bash
cp .env.example .env
nano .env
```

가장 간단한 Ollama 실행 예시는 다음 값입니다.

```dotenv
FDM_BACKEND=ollama
FDM_OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
FDM_MODEL_SMALL=qwen3:8b
FDM_MODEL_JUDGE=qwen3:8b
FDM_TIMEOUT=180
FDM_MAX_TOKENS=1200
FDM_THINK=0
FDM_KEEP_ALIVE=30m
```

vLLM 서버를 따로 띄우는 경우에는 다음처럼 바꿉니다.

```dotenv
FDM_BACKEND=vllm
FDM_VLLM_BASE_URL=http://127.0.0.1:8000/v1
FDM_MODEL_SMALL=Qwen/Qwen3-14B
FDM_MODEL_JUDGE=LGAI-EXAONE/EXAONE-4.0-32B
```

## 6. Ollama 백엔드 실행

Ollama를 설치합니다.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

모델을 내려받습니다.

```bash
ollama pull qwen3:8b
```

서버 프로세스가 떠 있는지 확인합니다.

```bash
curl http://127.0.0.1:11434/api/tags
```

응답이 오면 준비된 것입니다.

## 7. 기본 동작 확인

LLM 없이 파이프라인만 먼저 확인합니다.

```bash
FDM_BACKEND=mock uv run fdm doctor
```

이후 실제 LLM 백엔드로 확인합니다.

```bash
uv run fdm doctor
```

가벼운 명령으로 데이터 로딩도 확인합니다.

```bash
uv run fdm products
uv run fdm segments
```

## 8. CLI로 먼저 짧게 실행

처음에는 작은 설정으로 실행 시간을 확인합니다.

```bash
uv run fdm simulate 01_youth_step_saving --seeds 2 --personas-per-segment 2 --workers 2
```

서버 GPU 여유가 있으면 워커를 늘립니다.

```bash
uv run fdm simulate 01_youth_step_saving --seeds 3 --personas-per-segment 4 --workers 4 --with-sensitivity
```

결과는 `outputs/` 아래에 저장됩니다.

## 9. Streamlit UI 실행

### 권장: SSH 터널 방식

서버에서 `tmux` 세션을 엽니다.

```bash
tmux new -s fdm
```

서버 안에서 Streamlit을 실행합니다.

```bash
bash scripts/run_streamlit_remote.sh
```

기본값은 다음과 같습니다.

- host: `127.0.0.1`
- port: `8501`
- 외부 직접 접속: 닫힘

`tmux`에서 빠져나오려면 `Ctrl-b`를 누른 뒤 `d`를 누릅니다.

로컬 PC의 새 터미널에서 SSH 터널을 엽니다.

```bash
ssh -L 8501:127.0.0.1:8501 <USER>@<SERVER_IP>
```

브라우저에서 접속합니다.

```text
http://127.0.0.1:8501
```

### 선택: 공개 포트 방식

서버 보안 그룹이나 방화벽에서 `8501/tcp`를 허용한 뒤 실행합니다.

```bash
FDM_STREAMLIT_HOST=0.0.0.0 FDM_STREAMLIT_PORT=8501 bash scripts/run_streamlit_remote.sh
```

브라우저에서 접속합니다.

```text
http://<SERVER_IP>:8501
```

주의: Streamlit 자체에는 강한 인증 기능이 없습니다. 공개 포트 방식은 데모용으로만 쓰고, 장시간 운영은 SSH 터널, VPN, 또는 Nginx reverse proxy + 인증을 권장합니다.

## 10. 서버 프로세스 관리

실행 중인 `tmux` 세션으로 돌아가기:

```bash
tmux attach -t fdm
```

Streamlit 종료:

```bash
Ctrl-c
```

백그라운드에서 계속 돌리고 싶다면 `tmux`를 유지한 채 detach 합니다.

## 11. 자주 막히는 지점

### 브라우저에서 접속이 안 됨

SSH 터널 방식이면 로컬에서 터널 명령이 계속 실행 중인지 확인합니다.

```bash
ssh -L 8501:127.0.0.1:8501 <USER>@<SERVER_IP>
```

공개 포트 방식이면 서버 방화벽과 클라우드 보안 그룹에서 `8501/tcp`가 열려 있어야 합니다.

### LLM 연결 실패

Ollama 사용 시:

```bash
curl http://127.0.0.1:11434/api/tags
```

vLLM 사용 시:

```bash
curl http://127.0.0.1:8000/v1/models
```

`.env`의 `FDM_BACKEND`, `FDM_OLLAMA_BASE_URL`, `FDM_VLLM_BASE_URL` 값도 같이 확인합니다.

### GPU 메모리 부족

- `FDM_MODEL_SMALL`을 더 작은 모델로 낮춥니다.
- `--workers` 값을 `1` 또는 `2`로 낮춥니다.
- `FDM_MAX_TOKENS`를 `600` 정도로 낮춥니다.
- `FDM_THINK=0`을 유지합니다.

### 포트가 이미 사용 중

```bash
FDM_STREAMLIT_PORT=8502 bash scripts/run_streamlit_remote.sh
```

로컬 터널도 같은 포트로 맞춥니다.

```bash
ssh -L 8502:127.0.0.1:8502 <USER>@<SERVER_IP>
```
