# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**  
Sự đảo ngược thứ bậc giữa Train Loss và điểm Target thực tế khi so sánh `attn_only` ($r=283$) và `correct` ($r=16$). Dù cùng ngân sách 32.4 triệu tham số huấn luyện, `attn_only` có loss trên tập train thấp hơn hẳn (0.5371 so với 0.6267 của `correct`), nhưng khi đo độ chính xác mục tiêu thực tế lại thua (0.9375 so với 0.9688). Điều này chứng minh việc dồn rank vào chỉ 2 ma trận Attention ($q, v$) gây overfit cục bộ, và "Train Loss thấp" hoàn toàn có thể là một cái bẫy nếu không đo bằng benchmark độc lập.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**  
Tôi mất nhiều thời gian nhất ở khâu sinh văn bản kiểm thử (autoregressive generation / greedy decode) khi đo 3 mốc baseline và eval 4 cấu hình đối chứng (chiếm hơn 60% tổng thời gian pipeline). Ban đầu tôi dự đoán khâu huấn luyện forward/backward pass ở NB3 và NB4 sẽ là nút thắt cổ chai lớn nhất, nhưng thực tế việc sinh token tuần tự cho từng mẫu trên tập eval tốn nhiều thời gian chờ đợi hơn cả khâu huấn luyện.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**  
- Tôi từng tin rằng fine-tuning luôn là giải pháp bắt buộc đầu tiên, nhưng thực tế baseline (b) prompt tối ưu đã đạt 68.75% độ chính xác mà không tốn chi phí train hay rủi ro catastrophic forgetting.
- Tôi từng tin tăng Rank $r$ là đòn bẩy lớn nhất để nâng cao năng lực học của LoRA, nhưng thực tế vị trí gắn adapter toàn diện (`all-linear`) với $r=16$ áp đảo hoàn toàn $r=283$ ở attention-only.
- Tôi từng tin QLoRA 4-bit luôn là "viên đạn bạc" miễn phí, nhưng thực tế nó làm chậm tốc độ huấn luyện do chi phí giải lượng tử (dequantization) và làm tụt độ chính xác từ 96.88% xuống 84.38%.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**  
- **Việc đã dùng**: Dùng AI để tra cứu cơ chế hoạt động của Jinja chat template, công thức khớp ngân sách tham số `matched_rank`, và hỗ trợ rà soát các chỉ số so sánh trong báo cáo.
- **Chỗ AI sai**: AI ban đầu có xu hướng gợi ý cấu hình mặc định `bf16=True` cho `TrainingArguments` theo thói quen của các tutorial trên GPU Ampere/A100, trong khi GPU Colab T4 (Turing sm_75) không hỗ trợ native bfloat16 và bắt buộc phải dùng `fp16=True` kèm GradScaler. Ngoài ra, AI từng ngộ nhận lấy loss huấn luyện thấp của `attn_only` làm bằng chứng cho rằng cấu hình đó tốt hơn.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**  
Bước đầu tiên tôi làm KHÔNG PHẢI là nạp model hay chạy LoRA ngay, mà là:
1. Đóng băng một tập dữ liệu kiểm định độc lập (Eval/Test set) đại diện đúng cho phân phối thực tế của khách hàng, sau đó xây dựng Prompt tối ưu nhất có thể để đo mốc Baseline (b).
2. Nếu Baseline (b) chưa đạt yêu cầu nghiệp vụ và bắt buộc phải fine-tune, bước kỹ thuật đầu tiên là kiểm chứng **Loss Mask** (`assistant-only`) bằng cách decode ngược mảng token ID có nhãn $\ne -100$, đảm bảo 100% prompt và câu hỏi của người dùng không bị tính loss trước khi cấp quyền chạy GPU.
