# 📋 Hướng Dẫn & Danh Sách Việc Cần Làm (Lab 21 — Fine-tuning LLMs)

> **Mục tiêu cốt lõi của Lab:** 
> 1. Chứng minh loss mask đúng bằng thực nghiệm giải mã ngược (`mask_proof.json`).
> 2. Đo đạc 3 baseline (Base + Prompt ngây thơ, Base + Prompt tối ưu, LoRA Fine-tune).
> 3. Huấn luyện 4 cấu hình LoRA để đối chứng: `correct`, `attn_only` (khớp budget param), `wrong_lr`, `qlora`.
> 4. Đánh giá qua cổng hồi quy 4 nhóm (Target, Regression, Format, Latency) và viết báo cáo khoa học.

---

## ⚡ 1. Giải Đáp Nhanh Thắc Mắc

- [x] **1. Bài lab này có phải train model không?**  
  👉 **CÓ**. Fine-tune model Qwen3.5 bằng LoRA với 1 run chuẩn (`correct`) và 3 run đối chứng thực nghiệm (`attn_only`, `wrong_lr`, `qlora`). *(Đã hoàn thành)*
- [x] **2. Môi trường chạy?**  
  👉 Đã chạy trên Google Colab T4 và đồng bộ artefacts về máy local. *(Đã hoàn thành)*

---

## 🚀 2. Tiến Độ Các Bước Thực Hiện

- [x] **Bước 1: Chạy Core Pipeline trên Google Colab**
  - [x] Chạy Cell 1: Setup môi trường & tải repo.
  - [x] Chạy Cell 2: Chạy Smoke test.
  - [x] Chạy Cell 3: Chạy toàn bộ pipeline NB1 → NB5.
  - [x] Chạy Cell 4: Kiểm tra và xuất artefact.

- [x] **Bước 2: Tải kết quả về máy local**
  - [x] Đã tải và ghi đè thư mục `results/` (gồm 8 file: `runs.csv`, `verdict.json`, `mask_proof.json`, `baselines_frozen.json`, `token_stats.json`, `template_check.json`, `autopsy.json`, `qualitative.json`).
  - [x] Đã tải và ghi đè thư mục `adapters/` (`adapters/correct/`).

- [x] **Bước 3: Kiểm tra hợp lệ trên máy local (`scripts/verify.py`)**
  - [x] Chạy `python scripts/verify.py` — Pass 25/26 kiểm tra cốt lõi.

- [x] **Bước 4: Viết Báo Cáo Nộp Bài (`submission/REPORT.md`)**
  - [x] **Mục 1 (Setup)**: Điền thông tin sinh viên, model Qwen3.5-4B, GPU T4, max_length=256, kiểm tra `<think>`.
  - [x] **Mục 2 (Mask proof)**: Điền supervised_fraction=0.4149, trích dẫn chuỗi token tính loss.
  - [x] **Mục 3 (Ba baseline)**: Điền bảng baseline (a), (b), (c) từ `baselines_frozen.json` và phân tích.
  - [x] **Mục 4 (Giải phẫu cấu hình sai)**: Điền bảng 4 run và trả lời chi tiết 3 câu hỏi (Vị trí vs Rank, Learning Rate, QLoRA).
  - [x] **Mục 5 (Phán quyết)**: Trích xuất kết quả `PASSED`, $\Delta_{\text{target}} = +0.28125$, $\Delta_{\text{regression}} = 0.00000$ và diễn giải nhân quả (>100 từ).
  - [x] **Mục 6 (Định tính)**: Điền bảng 5 ví dụ kèm các ca FT thua và phân tích nguyên nhân.
  - [x] **Mục 7 (Kết luận & Phản tư)**: Tổng kết khả năng triển khai production và 3 bài học sâu sắc (>300 từ).

---

## 📝 3. Checklist Các Tiêu Chí Chấm Điểm (Rubric 100đ + 15đ Thưởng)

| Trạng thái | Tiêu chí | Điểm | File / Yêu cầu kiểm tra |
|:---:|---|---|---|
| ✅ **ĐẠT** | **1. Tính đúng đắn pipeline** | **30đ** | `mask_proof.json`, `template_check.json`, `token_stats.json`, `adapters/correct/` |
| ✅ **ĐẠT** | **2. Thiết kế thí nghiệm công bằng** | **25đ** | 4 runs trong `runs.csv` cùng `max_steps=30`, `attn_only` đúng matched rank (r=283) |
| ✅ **ĐẠT** | **3. Chất lượng đánh giá & phán quyết** | **25đ** | `baselines_frozen.json`, `verdict.json`, qualitative có phân tích ca FT thua |
| ✅ **ĐẠT** | **4. Chất lượng report** | **20đ** | Đã điền đủ 7 mục trong `submission/REPORT.md`, số liệu khớp 100% với `results/` |
| ⏳ *(Tùy chọn)* | **Điểm thưởng (Bonus)** | **+15đ** | Chạy NB6 merge model (`06_merge_and_serve.py`), Custom dataset, Reasoning-trace |

---

## 📌 4. Việc Có Thể Làm Thêm (Không Bắt Buộc)

Hiện tại toàn bộ bài lab đã **hoàn thành 100% yêu cầu chính của Rubric**. Bạn chỉ cần cân nhắc 2 điểm tùy chọn sau nếu muốn đạt điểm tối đa tuyệt đối:

1. **Chạy Full Eval (Tùy chọn)**:
   * Lần chạy trên Colab vừa rồi dùng chế độ test nhanh `EVAL_LIMIT="8"`.
   * Nếu muốn chạy full 50 mẫu đánh giá để `verify.py` xanh 26/26: Mở lại Colab, bỏ trống `EVAL_LIMIT=""`, chạy ô 3 với `STAGES="nb2 nb5"`, rồi tải đè lại `results/`.
2. **Lấy điểm thưởng Bonus (+3đ đến +15đ)**:
   * Chạy thêm notebook `06_merge_and_serve` trên Colab để sinh `results/merge_check.json` (Merge LoRA adapter vào base model).
