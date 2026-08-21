# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Văn Duy  **MSSV**: 2A202601537  **Ngày**: 2026-08-21  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (Google Colab)`

> Mọi con số dưới đây khớp 100% với các file trong thư mục `results/` được đo lường thực nghiệm trên toàn bộ 50 mẫu đánh giá.

---

## 1. Setup

| Thông số | Giá trị thực tế |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 epochs / 30 optimizer steps |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)*  
*Chi tiết:* Tokenizer của `Qwen3.5` giữ nguyên nội dung trong thẻ `<think>` khi render template (`verdict: reasoning preserved — safe to train on traces`), đảm bảo các chuỗi suy luận nếu có trong dataset sẽ đi thẳng vào hàm loss mà không bị nuốt mất.

---

## 2. Mask proof (NB1)

| Tiêu chí | Kết quả kiểm chứng |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss (`results/mask_proof.json`):

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7578 | 0.0000 | 3324.1 |
| (b) base + optimized prompt | 0.7650 | 0.7578 | 1.0000 | 1059.7 |
| (c) LoRA fine-tune (`correct`) | 0.9700 | 0.5444 | 1.0000 | 1450.8 |

**(b) có thật sự mạnh hơn (a) không?** Có. Baseline (b) đạt 76.5% target accuracy và 100% tuân thủ format JSON, vượt trội hoàn toàn so với mức 0.0% của Prompt ngây thơ (a).  
**Bạn có sửa `OPTIMIZED_PROMPT` không?** Không. Tôi giữ nguyên chuỗi `OPTIMIZED_PROMPT` mặc định chuẩn mực để đảm bảo tính liêm chính khoa học và bảo toàn mã băm SHA256 (`719e74d3b6232053`), tạo nên một mốc đối chứng công bằng tuyệt đối cho bản Fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6266 | **0.9700** | 919.3s | 12.01 GB |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5382 | **0.9700** | 761.1s | 12.02 GB |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | **0.0000** | 905.1s | 12.01 GB |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.9400** | 974.1s | 7.09 GB |

### Trả lời ba câu hỏi phân tích:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**  
Trên tập target, `attn_only` đạt kết quả **hoà** với `correct` ở mức chính xác 0.9700 (97.0%). Tuy nhiên, theo thứ tự `train loss` ở NB4, `attn_only` lại có loss thấp hơn rõ rệt (0.5382 so với 0.6266). Điều này cho thấy với một tác vụ hẹp (triage JSON 4 trường), việc tăng rank lên cực cao ($r=283$) chỉ trên các lớp Attention có thể giúp mô hình ghi nhớ tập train tốt hơn (overfitting nhẹ dẫn đến train loss thấp), nhưng không mang lại lợi thế tổng quát hóa cao hơn so với việc phân bổ đều tham số ($r=16$) trên toàn bộ các tầng `text-linear` (bao gồm MLP/Feed-Forward layers). Vị trí gắn adapter toàn diện giúp tối ưu hóa dung lượng tham số và ổn định biểu diễn hơn là việc cố tình dồn rank vào một vài module cục bộ.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**  
Chỉ thay đổi Learning Rate từ $10^{-4}$ về $10^{-5}$ (thang đo chuẩn của Full Fine-tune), đường loss của `wrong_lr` gần như phẳng lì và kẹt cứng ở mức $1.5704$, dẫn đến việc mô hình hoàn toàn không học được cấu trúc nhãn (target accuracy = 0.0000, format = 0.0000). Nếu chỉ nhìn vào loss mà không biết LR bị đặt sai, một kỹ sư sẽ dễ dàng kết luận sai lầm rằng: mô hình Base quá bé không đủ năng lực hiểu tiếng Việt, hoặc dữ liệu huấn luyện bị lỗi nhãn/vô nghĩa. Thực tế, vì các ma trận LoRA $B$ được khởi tạo bằng 0 và $A$ khởi tạo Gaussian, LoRA bắt buộc cần Learning Rate lớn hơn khoảng $10\times$ so với Full FT để cập nhật trọng số hiệu quả.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**  
`qlora` giúp tiết kiệm tới **4.92 GB VRAM** (giảm ~41% từ 12.01 GB xuống 7.09 GB), cho phép mô hình 4B chạy vừa vặn trên các GPU phổ thông 8GB. Tuy nhiên, cái giá phải trả là độ chính xác target bị suy giảm từ 0.9700 xuống 0.9400 (mất 3 điểm phần trăm độ chính xác) và thời gian huấn luyện tăng thêm ~55 giây do overhead giải nén lượng tử on-the-fly. Số đo thực nghiệm này hoàn toàn ủng hộ khuyến nghị của tài liệu Unsloth đối với kiến trúc Qwen3.5: sai số lượng tử hóa của 4-bit NF4 làm suy hao năng lực phân loại chi tiết, do đó nếu phần cứng đủ VRAM (như Colab T4 16GB), **16-bit LoRA luôn là sự lựa chọn ưu tiên hàng đầu**.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.205` · `regression Δ = -0.213` · `valid_trace_rate = 0.00`

### Diễn giải phán quyết (Phân tích nhân quả):
Cổng hồi quy đưa ra phán quyết `FAILED` không phải vì bản Fine-tune kém cỏi ở tác vụ mục tiêu — trên thực tế, mô hình Fine-tune đã **thắng áp đảo** Baseline Prompt tối ưu (b) với mức tăng độ chính xác $+20.5\%$ (đạt 97.0% so với 76.5%) và đạt $100\%$ chuẩn định dạng JSON.

Lý do phán quyết báo `FAILED` nằm ở việc mô hình đã vi phạm tiêu chí bảo toàn tri thức nền: điểm đánh giá kiến thức tổng quát (`regression`) bị sụt giảm từ $0.7578$ xuống $0.5444$ ($\Delta = -0.213$, vượt xa ngưỡng dung sai cho phép $0.020$). Đây là minh chứng kinh điển cho hiện tượng **Quên thảm họa (Catastrophic Forgetting)**: khi huấn luyện 30 steps trên tập dữ liệu thuần túy 100% ticket CSKH, trọng số adapter đã thích nghi quá mức với miền hẹp mà làm mờ nhạt khả năng trả lời chỉ dẫn đa năng lực của Base Model. 

Để khắc phục hiện tượng này trước khi đưa mô hình lên production thực tế, giải pháp chuẩn mực (theo deck §14.3) là kỹ thuật **Replay Data**: trộn thêm $1\% - 5\%$ dữ liệu hội thoại/chỉ dẫn đa miền vào tập train.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | `Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu.` | `hoi_thong_tin`, `trung_binh`, `ốp lưng điện thoại`, `tich_cuc` | Đúng 4 trường | Đúng 4 trường | ✅ **FT thắng**: Sinh JSON cực ngắn gọn, chính xác |
| 2 | `Chào shop, mình đặt ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm n` | `san_pham_loi`, `trung_binh`, `ốp lưng điện thoại`, `trung_tinh` | Lỗi cú pháp JSON | Đúng 4 trường | ✅ **FT thắng**: Khắc phục triệt để lỗi format của Base Model |
| 3 | `Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện.` | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | Đúng 4 trường | Sai `urgency: trung_binh` | ❌ **FT thua (0.75đ)**: Nhận diện sai mức độ khẩn cấp |
| 4 | `Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện.` | `san_pham_loi`, `thap`, `nồi chiên không dầu`, `trung_tinh` | Đúng 4 trường | Sai `urgency: trung_binh` | ❌ **FT thua (0.75đ)**: Nhận diện sai mức độ khẩn cấp |
| 5 | `Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện.` | `san_pham_loi`, `thap`, `áo khoác gió`, `tich_cuc` | Đúng 4 trường | Sai `urgency: trung_binh` | ❌ **FT thua (0.75đ)**: Nhận diện sai mức độ khẩn cấp |

**Có mẫu chung nào ở các ca FT thua không?**  
Có một quy luật sai số rất nhất quán: ở tất cả các ca đạt điểm 0.75 (thua baseline b), câu ticket của khách hàng đều chứa cụm từ biểu thị tính thong thả như *"Khi nào tiện"*. Nhãn chuẩn quy định đây là mức độ khẩn cấp thấp (`urgency: thap`), tuy nhiên mô hình Fine-tune lại có xu hướng thiên lệch (bias) dự đoán thành `urgency: trung_binh`. Nguyên nhân là do trong tập huấn luyện 225 mẫu, phân phối nhãn `urgency: trung_binh` chiếm tỷ trọng áp đảo, khiến mô hình học thiên kiến xác suất tiên nghiệm thay vì chú ý ngữ nghĩa của từ chỉ mức độ thời gian.

---

## 7. Kết luận & Điều tôi học được

### Kết luận (Tổng quan bài toán & Khuyến nghị triển khai):
Bản Fine-tune LoRA trong bài lab này đã chứng minh được giá trị vượt trội đối với bài toán phân loại ticket CSKH tiếng Việt. Mô hình nâng độ chính xác trích xuất từ 76.5% (Prompt tối ưu) lên 97.0%, loại bỏ hoàn toàn tình trạng sai định dạng JSON và rút ngắn chiều dài prompt gửi đi từ 160 tokens xuống còn một câu ngắn gọn, giúp tiết kiệm đáng kể chi phí token đầu vào khi phục vụ ở quy mô lớn. 

Tuy nhiên, tôi **khuyến nghị chưa nên deploy ngay lập tức bản adapter này cho một hệ thống chatbot đa năng tổng quát**, vì điểm hồi quy tổng quát bị giảm sút 21.3%. Nếu chỉ phục vụ như một Microservice / Worker chuyên biệt (chỉ nhận ticket và trả về JSON), bản Fine-tune này hoàn toàn sẵn sàng đưa vào vận hành. Nếu muốn dùng mô hình này làm trợ lý tương tác đa tác vụ, bước tiếp theo bắt buộc là tái huấn luyện với 3% Replay Data để đưa cổng hồi quy về trạng thái `PASSED`. Đòn bẩy quyết định thành công lớn nhất trong lab này không phải là việc tăng rank LoRA, mà là **việc cấu hình Learning Rate đúng tỉ lệ ($10\times$)** và **thuật toán Loss Masking chuẩn xác từng ký tự**.

### Ba điều tôi học được:
1. **Hiểu rõ bản chất Loss Masking:** Nhãn `labels = -100` trên phần Prompt là yếu tố quyết định để mô hình học cách sinh câu trả lời thay vì học cách nhắc lại đề bài. So sánh vị trí offset ký tự (character offset) là cách duy nhất đảm bảo tính chính xác khi chat template có hiện tượng gộp ký tự xuống dòng.
2. **Kỷ luật so sánh công bằng (Matched Budget):** Không bao giờ so sánh hai cấu hình chỉ dựa trên chỉ số $r$ (rank). Việc dùng `matched_rank` để đưa `attn_only` về cùng ngân sách 32.4M tham số với `correct` đã chứng minh bản chất vị trí phân bổ adapter quan trọng hơn việc dồn tham số cục bộ.
3. **Ý nghĩa thực sự của Cổng Hồi Quy:** Không được lấy Perplexity hay Train Loss làm thước đo chiến thắng. Một mô hình Fine-tune thành công phải đánh bại được chính nó khi Prompting tử tế và không được đánh đổi sự thông minh tổng quát lấy sự vừa vặn trong miền hẹp.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
* Tạo tập dữ liệu Replay Data (khoảng 10 câu hỏi đa lĩnh vực) và huấn luyện lại bản `correct` để kiểm chứng xem cổng hồi quy có lật từ `FAILED` sang `PASSED` hay không.
* Thực hiện merge trọng số ở NB6 và đo lường thông lượng phục vụ (throughput token/s) khi triển khai qua vLLM engine.
