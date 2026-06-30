# Hermes-Omni Reasoning Protocol — Push Bundle

Bundle này chứa toàn bộ code reasoning-first protocol (v1 + v1.1) mà Super Z
đã develop cho repo `airdropp20208-star/hermes-omni`.

## Nội dung bundle

```
hermes-omni-reasoning-push.zip
├── README.md                          ← file này (hướng dẫn chi tiết)
├── patches/                           ← CÁCH 1: git patches (giữ commit history)
│   ├── 0001-feat-reasoning-...-v1.patch       (108 KB, 2737 dòng)
│   └── 0002-feat-reasoning-v1.1-...patch       (85 KB, 2117 dòng)
└── files/                             ← CÁCH 2: raw files (mày tự commit)
    ├── agent/
    │   ├── omni_integration.py           (sửa — thêm deprecation notice)
    │   └── unified/
    │       ├── config.py                 (sửa — +13 config fields)
    │       ├── decision.py               (MỚI — DecisionFramework)
    │       ├── integration.py            (sửa — wire reasoning + longrun + router)
    │       ├── longrun.py                (MỚI — LongRunEngine + checkpoint)
    │       ├── reasoning.py              (MỚI — ReasoningProtocol + batch)
    │       ├── smart_guardian.py         (MỚI — LLM-as-judge guardian)
    │       └── tool_router.py            (MỚI — auto tool selection)
    ├── tools/
    │   └── reasoning_tools.py            (MỚI — 4 explicit reasoning tools)
    └── docs/
        └── REASONING_DESIGN.md           (MỚI — design doc đầy đủ)
```

## 2 commit sẽ được push

```
f09a2ba feat(reasoning): v1.1 — long-run engine + tool router   (+1,891 dòng)
47819a0 feat(reasoning): v1 — reasoning-first protocol          (+2,488 dòng)
0d1038f feat: unlimited mode part 2                              (upstream, đã có)
```

**Tổng cộng +4,379 dòng code mới**, tất cả opt-in (default OFF), legacy
behavior không đổi.

---

# HƯỚNG DẪN PUSH — ĐỌC KỸ TRƯỚC KHI LÀM

## ⚠️ Bước 0: REVOKE token cũ (BẮT BUỘC)

Token `ghp_***REDACTED***` đã bị dán vào chat 2 lần
→ coi như đã lộ vĩnh viễn. Phải revoke ngay:

1. Mở https://github.com/settings/tokens
2. Tìm token có prefix `ghp_bANFAY9...`
3. Click **Delete** (hoặc **Revoke**)
4. Xác nhận

Nếu mày không revoke, bất kỳ ai đọc chat log đều có thể push code độc
vào repo mày (và mọi repo khác mày own, vì classic token có scope rộng).

---

## Bước 1: Tạo token mới (Fine-grained, an toàn)

**Đừng dùng classic token nữa.** Dùng Fine-grained token để giới hạn scope:

1. Mở https://github.com/settings/personal-access-tokens/new
   (hoặc từ Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token)

2. Điền:
   - **Token name:** `hermes-omni-push` (hoặc gì cũng được)
   - **Expiration:** 7 days (hoặc ít hơn — chỉ cần push 1 lần)
   - **Repository access:** **Only select repositories** → chọn
     `airdropp20208-star/hermes-omni`
   - **Permissions → Repository permissions:**
     - **Contents:** Read and write (để push code)
     - **Metadata:** Read (tự động được bật, bắt buộc)

3. Click **Generate token**
4. **COPY token ngay** (chỉ hiển thị 1 lần). Format sẽ là `github_pat_...`
   (không phải `ghp_...`).

5. **KHÔNG dán token này vào chat.** Paste vào notepad tạm, hoặc giữ
   trong clipboard. Mày sẽ paste trực tiếp vào `git push` ở bước sau.

---

## Bước 2: Giải nén bundle

Tải file `hermes-omni-reasoning-push.zip` về máy mày (từ
`/home/z/my-project/download/`), rồi:

```bash
# Tạo thư mục làm việc
mkdir -p ~/hermes-push
cd ~/hermes-push

# Copy zip vào (hoặc download về)
# Giả sử zip nằm ở ~/Downloads/hermes-omni-reasoning-push.zip
unzip ~/Downloads/hermes-omni-reasoning-push.zip

# Verify
ls -la
# Phải thấy: README.md, patches/, files/
```

---

## Bước 3: Clone repo về máy

```bash
cd ~/hermes-push
git clone https://github.com/airdropp20208-star/hermes-omni.git
cd hermes-omni

# Verify đang ở branch main
git branch
# → * main

# Verify upstream commit mới nhất
git log --oneline -3
# Phải thấy:
# 0d1038f feat: unlimited mode part 2 — remove remaining limits
# be3646d feat: unlimited mode — remove all agent limits
# 99097a0 feat: integrate OmniAgent features into conversation loop
```

---

## Bước 4: CHỌN 1 trong 2 cách apply code

### 🟢 CÁCH 1 (KHUYẾN NGHỊ): Apply git patches

**Ưu điểm:** Giữ nguyên commit history, commit message, author. Đẹp và
sạch. Push lên GitHub sẽ thấy 2 commit riêng biệt với message đầy đủ.

```bash
cd ~/hermes-push/hermes-omni

# Apply patch 1 (v1 — reasoning protocol)
git am ../patches/0001-*.patch
# Output mong đợi:
# Applying: feat(reasoning): add reasoning-first protocol v1 — plan/critique/execute/reflect

# Apply patch 2 (v1.1 — longrun + tool router)
git am ../patches/0002-*.patch
# Output mong đợi:
# Applying: feat(reasoning): v1.1 — long-run engine + tool router

# Verify
git log --oneline -5
# Phải thấy:
# f09a2ba feat(reasoning): v1.1 — long-run engine + tool router
# 47819a0 feat(reasoning): add reasoning-first protocol v1 — plan/critique/execute/reflect
# 0d1038f feat: unlimited mode part 2 — remove remaining limits
# be3646d feat: unlimited mode — remove all agent limits
# 99097a0 feat: integrate OmniAgent features into conversation loop

# Verify files tồn tại
ls agent/unified/
# Phải thấy: config.py  decision.py  events.py  integration.py  memory_provider.py
#            policy.py  reasoning.py  reflexion.py  smart_guardian.py  tracing.py
#            tool_router.py  longrun.py  __init__.py

ls tools/reasoning_tools.py
# Phải thấy file tồn tại

ls docs/REASONING_DESIGN.md
# Phải thấy file tồn tại
```

**Nếu `git am` báo lỗi** (rất hiếm khi xảy ra vì patches được tạo từ
repo gốc), xem section "Khắc phục sự cố" bên dưới.

---

### 🟡 CÁCH 2: Copy raw files + tự commit

**Khi nào dùng:** Cách 1 fail, hoặc mày muốn gộp thành 1 commit duy nhất.

```bash
cd ~/hermes-push/hermes-omni

# Copy toàn bộ files từ bundle vào repo (overwrite nếu trùng)
cp -r ../files/* .

# Verify
ls agent/unified/longrun.py        # file mới
ls agent/unified/tool_router.py    # file mới
ls agent/unified/reasoning.py      # file mới
ls tools/reasoning_tools.py        # file mới
ls docs/REASONING_DESIGN.md        # file mới

# Check git status — phải thấy 10 file changed
git status
# Expected:
# modified:   agent/omni_integration.py
# modified:   agent/unified/config.py
# modified:   agent/unified/integration.py
# new file:   agent/unified/decision.py
# new file:   agent/unified/longrun.py
# new file:   agent/unified/reasoning.py
# new file:   agent/unified/smart_guardian.py
# new file:   agent/unified/tool_router.py
# new file:   docs/REASONING_DESIGN.md
# new file:   tools/reasoning_tools.py

# Stage tất cả
git add -A

# Commit
git commit -m "feat(reasoning): v1+v1.1 — reasoning-first protocol, long-run engine, tool router

- DecisionFramework: classify actions by consequence (TRIVIAL/STANDARD/CONSEQUENTIAL/IRREVERSIBLE)
- ReasoningProtocol: plan → critique → execute → reflect (single LLM call per phase)
- SmartGuardian: LLM-as-judge with cache + TTL, hard floor for IRREVERSIBLE
- LongRunEngine: priority queue + checkpoint/resume + background reflection worker
- ToolRouter: BM25 + 25 intent patterns + usage learning
- 4 explicit reasoning tools: reasoning_plan/critique/decide/reflect
- All opt-in via unified.reasoning/smart_guardian/longrun/tool_router config
- Default OFF, legacy behavior unchanged
- See docs/REASONING_DESIGN.md for full design"
```

---

## Bước 5: Push lên GitHub

```bash
cd ~/hermes-push/hermes-omni

# Push
git push origin main
```

**Git sẽ hỏi:**
```
Username for 'https://github.com': airopp20208-star
Password for 'https://airdropp20208-star@github.com': [PASTE TOKEN VÀO ĐÂY]
```

- **Username:** `airdropp20208-star` (tên GitHub của mày)
- **Password:** **PASTE TOKEN MỚI** (`github_pat_...`) vào — KHÔNG phải
  password GitHub của mày. Khi paste, terminal sẽ không hiển thị gì (an
  toàn), cứ paste rồi Enter.

**Nếu thành công:**
```
Enumerating objects: 25, done.
Counting objects: 100% (25/25), done.
Delta compression using up to 8 threads
Compressing objects: 100% (15/15), done.
Writing objects: 100% (15/15), 196.05 KiB | 8.17 MiB/s, done.
Total 15 (delta 12), reused 0 (delta 0), pack-reused 0
To https://github.com/airdropp20208-star/hermes-omni.git
   0d1038f..f09a2ba  main -> main
```

---

## Bước 6: Verify trên GitHub

1. Mở https://github.com/airdropp20208-star/hermes-omni
2. Check commit history — phải thấy 2 commit mới (hoặc 1 nếu dùng Cách 2)
3. Click vào file `docs/REASONING_DESIGN.md` để verify nội dung
4. Click vào `agent/unified/` để verify có `longrun.py`, `tool_router.py`,
   `reasoning.py`, `smart_guardian.py`, `decision.py`

---

## Bước 7: Dọn dẹp (sau khi push thành công)

```bash
# Xóa token khỏi clipboard / notepad ngay
# (trên macOS: pbcopy < /dev/null để clear clipboard)

# Revoke token trên GitHub (vì đã push xong, không cần nữa)
# Mở https://github.com/settings/personal-access-tokens
# Tìm token "hermes-omni-push" → Revoke

# Xóa bundle tạm (tùy chọn)
# rm -rf ~/hermes-push
```

---

# Khắc phục sự cố

## Lỗi: `git am` báo conflict

```
error: patch failed: agent/unified/integration.py:1
error: agent/unified/integration.py: patch does not apply
```

**Nguyên nhân:** Repo local của mày có thay đổi mà patch không biết.
Patches được tạo dựa trên commit `0d1038f` (upstream gốc).

**Khắc phục:**
```bash
# Reset về upstream sạch
git fetch origin
git reset --hard origin/main

# Thử lại
git am --abort 2>/dev/null
git am ../patches/0001-*.patch
git am ../patches/0002-*.patch
```

Nếu vẫn fail → dùng **Cách 2** (copy raw files).

## Lỗi: `git push` báo `Permission denied`

```
remote: Permission to airdropp20208-star/hermes-omni.git denied to <user>.
fatal: unable to access 'https://github.com/airdropp20208-star/hermes-omni.git/': The requested URL returned error: 403
```

**Nguyên nhân:** Token không có quyền Contents: Read and write, hoặc
mày paste nhầm username.

**Khắc phục:**
1. Kiểm tra token có quyền **Contents: Read and write** (Fine-grained)
2. Username phải là `airdropp20208-star` (chính xác, kể cả dấu gạch)
3. Thử lại `git push origin main`

## Lỗi: `git push` báo `non-fast-forward`

```
! [rejected]        main -> main (non-fast-forward)
```

**Nguyên nhân:** Có ai đã push lên repo kể từ khi mày clone.

**Khắc phục:**
```bash
git pull --rebase origin main
git push origin main
```

## Lỗi: Token classic `ghp_...` không hoạt động

Đúng — GitHub đã disable token này (hoặc mày đã revoke). Phải tạo
Fine-grained token mới theo Bước 1.

## Lỗi: Muốn push nhưng không muốn qua terminal

Mày có thể upload files trực tiếp qua web UI GitHub:
1. Mở https://github.com/airdropp20208-star/hermes-omni
2. Click **Add file → Upload files**
3. Drag & drop các file từ `files/` trong bundle (cần tạo subfolder
   `agent/unified/`, `tools/`, `docs/` thủ công)
4. Commit message: `feat(reasoning): v1+v1.1 — reasoning-first protocol`
5. Click **Commit changes**

⚠️ Cách này lâu hơn (10 file, có subfolder) và không giữ được commit
history đẹp. Chỉ dùng nếu terminal không khả thi.

---

# Tóm tắt nhanh (cho pro)

```bash
# 1. Revoke old token: https://github.com/settings/tokens
# 2. Create new fine-grained token: https://github.com/settings/personal-access-tokens/new
#    - Repo: airdropp20208-star/hermes-omni
#    - Permissions: Contents Read+Write, Metadata Read

# 3. On local machine:
mkdir ~/hermes-push && cd ~/hermes-push
unzip hermes-omni-reasoning-push.zip
git clone https://github.com/airdropp20208-star/hermes-omni.git
cd hermes-omni
git am ../patches/0001-*.patch
git am ../patches/0002-*.patch
git log --oneline -5  # verify 2 commits mới
git push origin main  # username=airdropp20208-star, password=token mới

# 4. Verify: https://github.com/airdropp20208-star/hermes-omni
# 5. Revoke new token (đã push xong)
```

---

# Liên hệ

Nếu gặp lỗi gì mà README này không cover, mô tả lỗi cho mình (copy
paste terminal output), mình sẽ debug giúp. Đừng dán token vào chat —
dán output lệnh git thôi là đủ.
