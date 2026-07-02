<div align="center">

# ⚡ Hermes-Omni

### Agent AI tự cải tiến — fork của Hermes v0.17.0

[![Dashboard](https://img.shields.io/badge/Dashboard-v6-c8860d?style=for-the-badge)](#-dashboard)
[![Modules](https://img.shields.io/badge/Modules-44-blueviolet?style=for-the-badge)](#-modules)
[![Security](https://img.shields.io/badge/Security-Auth%20%2B%20Rate%20Limit-green?style=for-the-badge)](#-bảo-mật)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**44 modules reasoning · Dashboard web full-featured · Chạy trên UserLAnd/Termux/Linux**

</div>

---

## 📑 Mục lục

- [✨ Tính năng](#-tính-năng)
- [🚀 Cài đặt nhanh](#-cài-đặt-nhanh)
- [🖥️ Dashboard](#️-dashboard)
- [🧠 Modules](#-modules)
- [🔒 Bảo mật](#-bảo-mật)
- [⚙️ Cấu hình](#️-cấu-hình)
- [📱 Chạy trên UserLAnd](#-chạy-trên-userland)
- [🔧 Khắc phục sự cố](#-khắc-phục-sự-cố)

---

## ✨ Tính năng

Hermes-Omni biến Hermes Agent thành agent reasoning-first sánh ngang Claude/GLM, với:

| Tính năng | Mô tả |
|-----------|-------|
| 🧠 **Reasoning pipeline** | Lập kế hoạch → đánh giá → thực hiện → rút kinh nghiệm |
| 🛡️ **Smart Guardian** | LLM đánh giá output trước khi gửi |
| 🔍 **Verifier** | Tự kiểm tra tính chính xác |
| 📜 **Constitution** | Nguyên tắc đạo đức tích hợp |
| 🐢 **Slow Thinking** | 4 mức suy luận: Fast / Balanced / Deep / Max |
| 🎯 **Cognitive Tree** | Cây suy nghĩ phân nhánh |
| 🔬 **Hypothesis** | Đặt giả thuyết + kiểm chứng |
| 🧬 **Causal Graph** | Đồ thị nhân quả |
| 📚 **Skill Registry** | 113 kỹ năng từ 10 repo (anthropics, claude...)
| 🔌 **API Registry** | 1500+ API công khai
| 💰 **Cost Tracker** | Đếm token + ngân sách
| 🗄️ **Response Cache** | Lưu cache tiết kiệm token
| 👤 **User Model** | Cá nhân hóa theo người dùng
| ❓ **Clarifier** | Phát hiện câu hỏi mơ hồ → hỏi lại
| 🔄 **Reflexion** | Học từ lỗi sai
| 📈 **Learning** | Đường cong quên (Ebbinghaus)
| 🛠️ **Skill Synthesizer** | Tự tạo kỹ năng mới
| 📋 **Task Planner** | Chia task phức tạp thành subtask

---

## 🚀 Cài đặt nhanh

### Yêu cầu

- Python 3.11+
- ~200MB disk space
- API key từ bất kỳ provider nào (GLM, OpenAI, Xiaomi MiMo...)

### Cài đặt

```bash
# Clone repo
git clone https://github.com/airdropp20208-star/hermes-omni.git
cd hermes-omni

# Cài deps
pip install -e .
pip install pyyaml openai httpx requests

# Khởi động dashboard
./start.sh
```

Hoặc chạy trực tiếp:

```bash
python -m agent.unified.dashboard_server --port 8788
```

Server in ra **token auth** trong terminal. Mở `http://localhost:8788`, nhập token để đăng nhập.

---

## 🖥️ Dashboard

Dashboard web full-featured với 7 tabs:

### 💬 Trò chuyện
- Chat realtime với agent
- Mode panel: Thinking + Reasoning + Verify
- Hiển thị thời gian xử lý
- Hỗ trợ kéo thả file
- Auto-refresh khi không chat

### 📊 Tổng quan
- Provider hiện tại + API key status
- Số skills đã cài
- Features đã bật (17/21)
- Token usage

### 🔌 Nhà cung cấp API
- Form setup 10 providers (GLM, OpenAI, Anthropic, OpenRouter...)
- Test key trực tiếp
- Multi-provider: add/remove/toggle key
- Sync `config.yaml` + `.env`

### ⚙️ Cấu hình
- 21 features toggle on/off
- Atomic write (không corrupt khi crash)
- Search filter

### 📚 Kỹ năng
- 113 skills từ 10 repos
- Search filter
- Auto-discovery

### 💰 Chi phí
- Token usage theo phase
- Bar chart trực quan
- Call count

### 📋 Nhật ký
- Activity log realtime
- Multiple log paths
- Auto-refresh 8s

---

## 🧠 Modules

44 modules trong `agent/unified/`:

### Nền tảng (v1)
```
reasoning.py          — Pipeline lập kế hoạch → đánh giá → thực hiện
reflexion.py          — Học từ lỗi sai
smart_guardian.py     — LLM đánh giá output
policy.py             — Chính sách an toàn
decision.py           — Ra quyết định
```

### Suy luận sâu (v2-v4)
```
cognitive_tree.py     — Cây suy nghĩ
hypothesis.py         — Giả thuyết + kiểm chứng
metacognitive.py      — Siêu nhận thức
causal_graph.py       — Đồ thị nhân quả
slow_thinking.py      — 4 mức suy luận
context_distiller.py  — Chưng cất context
context_hologram.py   — Hologram context
failure_forecast.py   — Dự báo lỗi
trajectory_distiller.py — Chưng cất quỹ đạo
skill_evolution.py    — Tiến hóa kỹ năng
persona_split.py      — Multi-persona
harness.py            — Test harness
```

### Hạ tầng (v3)
```
cost_tracker.py       — Đếm token
response_cache.py     — Cache response
streaming.py          — Streaming output
embedding.py          — Embedding cho memory
multi_provider.py     — Gộp nhiều API key
tool_router.py        — Tự chọn tool
longrun.py            — Tác vụ nền
```

### Thư viện (v3.2)
```
skill_registry.py     — 113 skills
api_registry.py       — 1500+ APIs
capability_resolver.py — Auto-install tools
```

### Học tập (v2.1-v3.1)
```
learning.py           — Đường cong quên
skill_synthesizer.py  — Tự tạo skill
user_model.py         — Model người dùng
clarifier.py          — Hỏi lại khi mơ hồ
task_planner.py       — Chia task
output_formatter.py   — Format Telegram/Slack
```

### Tích hợp
```
config.py             — Quản lý config
integration.py        — Wire modules
runtime_wiring.py     — 7 hooks vào mega-files
dashboard_server.py   — Web dashboard v6
```

---

## 🔒 Bảo mật

Dashboard v6 có bảo mật đầy đủ:

| Tính năng | Chi tiết |
|-----------|----------|
| 🔐 **Auth token** | Random 32 chars, lưu `~/.hermes/.dashboard_token` (chmod 600) |
| 🏠 **Bind localhost** | Mặc định `127.0.0.1` — không ai cùng WiFi truy cập được |
| 🚫 **DNS-rebinding guard** | Chỉ accept Host: localhost/127.0.0.1/::1 |
| ⏱️ **Rate limit** | 300 req/phút/IP (429 nếu exceed) |
| 📦 **Body limit** | 10MB max per request |
| 🛡️ **Path traversal** | `/api/download/` sanitize filename + resolve check |
| 🔑 **.env protection** | chmod 0600, atomic write |
| ✍️ **Atomic YAML** | Write `.tmp` + rename (không corrupt khi crash) |

### Truy cập từ máy khác (an toàn)

```bash
# SSH tunnel (khuyến nghị)
ssh -L 8788:localhost:8788 user@phone-ip

# Hoặc Tailscale
# Cài tailscale trên cả 2 máy, rồi truy cập http://phone-ip:8788
```

### Revoke token

```bash
python -m agent.unified.dashboard_server --new-token
```

---

## ⚙️ Cấu hình

### File `~/.hermes/config.yaml`

```yaml
model:
  provider: xiaomi          # zai, openai, anthropic, openrouter...
  default: mimo-v2.5        # model name
  base_url: https://api.xiaomimimo.com/v1

unified:
  reasoning:
    enabled: true
  verifier:
    enabled: true
  slow_thinking:
    enabled: true
    default_level: balanced  # fast, balanced, deep, max
  # ... 21 features tổng cộng
```

### File `~/.hermes/.env`

```bash
XIAOMI_API_KEY=sk-...
HERMES_INFERENCE_PROVIDER=xiaomi
HERMES_INFERENCE_MODEL=mimo-v2.5
HERMES_YOLO_MODE=1
HERMES_ACCEPT_HOOKS=1
```

### Providers hỗ trợ

| Provider | Base URL | Default Model |
|----------|----------|---------------|
| **z.ai (GLM)** | `https://open.bigmodel.cn/api/paas/v4` | glm-4.6 |
| **Xiaomi MiMo** | `https://api.xiaomimimo.com/v1` | mimo-v2.5 |
| **OpenAI** | `https://api.openai.com/v1` | gpt-4o-mini |
| **Anthropic** | `https://api.anthropic.com` | claude-3-5-sonnet |
| **OpenRouter** | `https://openrouter.ai/api/v1` | openai/gpt-4o-mini |
| **DeepSeek** | `https://api.deepseek.com/v1` | deepseek-chat |
| **Groq** | `https://api.groq.com/openai/v1` | llama-3.1-70b |
| **Together** | `https://api.together.xyz/v1` | llama-3-70b |
| **Mistral** | `https://api.mistral.ai/v1` | mistral-large |
| **Custom** | any | any |

---

## 📱 Chạy trên UserLAnd

### Yêu cầu

- UserLAnd app (Android)
- Ubuntu distro
- Python 3.11+

### Cài đặt

```bash
# Trong UserLAnd terminal
sudo apt update && sudo apt install -y python3 python3-pip git

git clone https://github.com/airdropp20208-star/hermes-omni.git
cd hermes-omni

pip install -e .
pip install pyyaml openai httpx requests

# Set API key
export XIAOMI_API_KEY=sk-...
echo "XIAOMI_API_KEY=sk-..." > ~/.hermes/.env

# Khởi động
./start.sh
```

### Truy cập từ browser điện thoại

```bash
# Dashboard chạy trên 127.0.0.1:8788
# Mở Chrome → http://localhost:8788
```

### Tips cho UserLAnd

- **Cold start chat:** 30-60s (do subprocess spawn)
- **Memory:** ~150MB server + ~500MB khi chat
- **Battery:** Chat tốn CPU, hạn chế chat dài
- **Network:** Bind localhost, dùng SSH tunnel nếu truy cập từ PC

---

## 🔧 Khắc phục sự cố

### Server không khởi động

```bash
# Check port có bị busy không
lsof -i :8788

# Kill process cũ
pkill -f dashboard_server

# Restart
./start.sh
```

### Chat không trả lời

```bash
# Check API key
cat ~/.hermes/.env

# Test key trực tiếp
curl -X POST https://api.xiaomimimo.com/v1/chat/completions \
  -H "Authorization: Bearer $XIAOMI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"mimo-v2.5","messages":[{"role":"user","content":"hi"}]}'
```

### Quên token

```bash
# Xem token
cat ~/.hermes/.dashboard_token

# Hoặc tạo token mới
python -m agent.unified.dashboard_server --new-token
```

### Browser không kết nối được

```bash
# Check server đang chạy
pgrep -f dashboard_server

# Check bind
ss -tlnp | grep 8788

# Restart
pkill -f dashboard_server && ./start.sh
```

### Log debug

```bash
# Server log
tail -f /tmp/dash*.log

# Chat log (in trong terminal)
# [chat] start: 'message'
# [chat] done in 32.5s, rc=0, out=1B, err=0B
```

---

## 📊 Kiến trúc

```
hermes-omni/
├── agent/
│   └── unified/
│       ├── dashboard_server.py    # Web dashboard v6
│       ├── reasoning.py           # Pipeline chính
│       ├── verifier.py            # Tự kiểm tra
│       ├── constitution.py        # Nguyên tắc
│       ├── slow_thinking.py       # Suy luận sâu
│       ├── skill_registry.py      # 113 skills
│       ├── ...                    # 44 modules
│       └── runtime_wiring.py      # 7 hooks
├── hermes_cli/                    # CLI (fork từ Hermes)
├── run_agent.py                   # AIAgent class
├── tools/                         # Agent tools
├── skills/local-repos/            # Skill repos
├── scripts/
│   ├── evaluate_cognitive.py      # Evaluation
│   └── clone-skills.sh            # Clone skills
├── start.sh                       # Launcher
└── install.sh                     # Installer
```

---

## 🤝 Đóng góp

Đây là fork của [Hermes Agent](https://github.com/NousResearch/hermes-agent) bởi Nous Research. Các modules bổ sung được phát triển riêng cho Hermes-Omni.

### Chạy evaluation

```bash
python scripts/evaluate_cognitive.py
# 100% pass cho 44 modules
```

---

## 📜 License

MIT License — xem [LICENSE](LICENSE).

<div align="center">

**⚡ Hermes-Omni — Agent AI tự cải tiến**

</div>
