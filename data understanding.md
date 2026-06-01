# Hiểu dữ liệu

```
1. Students are not allowed to: 
- Turn on Internet during the official Kaggle run. 
- Use external LLMs, APIs, online solvers, or additional answer-generation tools. 
- Use additional training data outside the provided train.json. 
- Replace the fixed base model with another model. 
- Put gold answers into prediction files. 
- Upload or manually edit test_predictions.json as a separate file. 
- Modify the scoring logic to misreport results. 
- Perform test-time learning or update the model on each test example. 
2. During inference, the model must remain frozen: 
- No .backward() 
- No .step() 
- No online adaptation on test samples
```

## 1. Bản chất dữ liệu

Dữ liệu là bài toán **Math Word Problems tiếng Việt** để fine-tune GPT-2 theo dạng causal language modeling: input là câu hỏi, model sinh tiếp lời giải và đáp án. Theo đề bài, chỉ nên dùng `query_vi` làm input và `response_vi` làm target; `type` dùng tốt cho phân tích lỗi/stratified evaluation, không nên đưa vào prompt nếu muốn bám sát yêu cầu “use only Vietnamese fields”. đề bài

Mỗi record có 7 field:

```
original_question_vi
original_question_en
query_vi
query_en
response_vi
response_en
type
```

Ví dụ dữ liệu có cả bài GSM dạng lời văn đời thường, bài MATH có LaTeX, và các biến thể như `AnsAug`, `Rephrased`, `SV`, `FOBAR`. Một số mẫu cho thấy `query_vi` có thể khác `original_question_vi`, ví dụ biến thể hỏi ngược để tìm biến `x`, nên **không được coi `original_question_vi` là input chính**. train

## 2. Thống kê chính từ `train.json`

Tôi parse được **100.000 mẫu**, không thiếu field và không có field rỗng.

| Thuộc tính                             | Kết quả              |
| -------------------------------------- | -------------------- |
| Tổng số mẫu                            | 100.000              |
| Số field / record                      | 7                    |
| Thiếu field                            | 0                    |
| Field rỗng                             | 0                    |
| Số `query_vi` unique                   | 58.925               |
| Số `original_question_vi` unique       | 13.101               |
| Exact duplicate query-response         | 1.032 dòng dư        |
| Query trùng nhưng đáp án conflict      | 106 query / 447 dòng |
| `query_vi` khác `original_question_vi` | 60.619 mẫu           |

Phân bố `type`:

| Type             | Số mẫu | Tỷ lệ gần đúng |
| ---------------- | ------ | -------------- |
| `GSM_AnsAug`     | 20.329 | 20.3%          |
| `GSM_Rephrased`  | 20.298 | 20.3%          |
| `MATH_AnsAug`    | 18.991 | 19.0%          |
| `MATH_Rephrased` | 12.788 | 12.8%          |
| `GSM_FOBAR`      | 10.191 | 10.2%          |
| `GSM_SV`         | 9.920  | 9.9%           |
| `MATH_FOBAR`     | 3.769  | 3.8%           |
| `MATH_SV`        | 3.714  | 3.7%           |

Gộp theo nguồn bài:

| Nhóm | Số mẫu |
| ---- | ------ |
| GSM  | 60.738 |
| MATH | 39.262 |

Gộp theo kiểu augmentation:

| Kiểu      | Số mẫu |
| --------- | ------ |
| AnsAug    | 39.320 |
| Rephrased | 33.086 |
| FOBAR     | 13.960 |
| SV        | 13.634 |

Nhận xét: dữ liệu không chỉ là “câu hỏi gốc → đáp án gốc”, mà là tập đã augmentation mạnh. Vì vậy preprocessing phải tôn trọng `query_vi`, không tự thay bằng `original_question_vi`.

## 3. Độ dài dữ liệu

| Trường                  | Min | Median | Mean  | P90  | P95  | P99  | Max  |
| ----------------------- | --- | ------ | ----- | ---- | ---- | ---- | ---- |
| `query_vi` chars        | 10  | 196    | 212.3 | 358  | 418  | 575  | 2310 |
| `response_vi` chars     | 14  | 407    | 475.6 | 816  | 974  | 1442 | 3444 |
| prompt + response chars | 50  | 635    | 707.9 | 1149 | 1340 | 1853 | 3841 |
| `query_vi` words        | 2   | 46     | 48.6  | 83   | 96   | 126  | 311  |
| `response_vi` words     | 4   | 97     | 113.5 | 199  | 237  | 336  | 772  |
| prompt + response words | 10  | 149    | 166.0 | 277  | 320  | 422  | 850  |

Tỷ lệ mẫu dài:

| Điều kiện                      | Số mẫu | Tỷ lệ  |
| ------------------------------ | ------ | ------ |
| prompt + response > 1000 chars | 16.871 | 16.87% |
| > 1500 chars                   | 2.847  | 2.85%  |
| > 2000 chars                   | 652    | 0.65%  |
| > 2500 chars                   | 199    | 0.20%  |
| > 3000 chars                   | 51     | 0.05%  |

Kết luận: phần lớn dữ liệu đủ ngắn để fine-tune với `MAX_LENGTH=512`. Tuy nhiên, nhóm MATH có một số câu rất dài, chứa LaTeX hoặc `[asy]` diagram code. Nếu truncate không cẩn thận, có thể mất phần đáp án cuối.

## 4. Đặc tính đáp án

Các response thường kết thúc bằng một anchor đáp án:

| Pattern                             | Số mẫu |
| ----------------------------------- | ------ |
| Có `Đáp án là:`                     | 74.105 |
| Có `Câu trả lời là:`                | 25.532 |
| Có một trong hai anchor tiếng Việt  | 99.637 |
| Anchor tiếng Việt gần cuối response | 99.636 |
| Có `####`                           | 59.762 |
| Có `\boxed{...}`                    | 29.356 |

Loại đáp án cuối trích được:

| Loại đáp án                                | Số mẫu |
| ------------------------------------------ | ------ |
| Số nguyên plain                            | 91.713 |
| LaTeX fraction, ví dụ `\frac{1}{48}`       | 3.455  |
| Có text/symbolic, ví dụ `50\sqrt{10}`, `n` | 2.388  |
| Decimal dùng dấu phẩy                      | 982    |
| Decimal dùng dấu chấm                      | 558    |
| Base subscript, ví dụ `2104_5`             | 53     |
| Không trích được bằng parser đơn giản      | 203    |

Kết luận: đa số là số nguyên, nhưng không thể preprocessing theo hướng “chỉ giữ số nguyên”. Nhóm MATH có phân số, căn, cơ số, biểu thức LaTeX. Nếu ép hết về số thập phân sẽ làm hỏng một phần target.

## 5. Nhiễu và vấn đề dữ liệu đáng chú ý

Có một số nhóm nhiễu rõ:

| Vấn đề                                     | Số mẫu gần đúng | Ý nghĩa                                              |
| ------------------------------------------ | --------------- | ---------------------------------------------------- |
| `query_vi` khác `original_question_vi`     | 60.619          | Do augmentation; không được thay query bằng original |
| Query có biến chưa biết `x/X`              | 35.683          | Nhiều bài dạng tìm biến ngược                        |
| Có phrase “Giá trị của biến ... chưa biết” | 27.446          | Đặc trưng của SV/FOBAR                               |
| Lặp câu “Giá trị của biến ...”             | 3.201           | Nhiễu text có thể clean                              |
| Có LaTeX/math syntax                       | 33.685          | Cần giữ ký hiệu toán                                 |
| Có decimal comma                           | 17.173          | Cần xử lý nhất quán khi extract đáp án               |
| Có `[asy]` diagram code                    | 1.012           | Rất dài, dễ gây nhiễu                                |
| Có `\đóng hộp{...}`                        | 752             | Lỗi dịch `\boxed`                                    |
| Có English leakage trong query/response    | khoảng vài trăm | Nên clean nhẹ hoặc loại mẫu nặng                     |
| Exact duplicate query-response             | 1.032 dòng dư   | Có thể drop                                          |
| Query trùng nhưng đáp án khác              | 447 dòng        | Cần kiểm tra/drop để tránh học mâu thuẫn             |

Điểm quan trọng: **không deduplicate theo `original_question_vi`**, vì một bài gốc có thể được biến đổi thành nhiều câu hỏi khác nhau, target khác nhau. Ví dụ cùng bài gốc có thể có bản hỏi đáp án cuối và bản hỏi ngược tìm biến `x`. Dedup theo original sẽ làm mất dữ liệu hợp lệ hoặc gây sai target.

## 6. Kết luận ngắn gọn

Dữ liệu khá lớn, sạch về schema nhưng nhiễu về format. Điểm quan trọng nhất là **chuẩn hóa target output**, không phải clean ngôn ngữ quá mạnh. Hướng dự kiến là giữ `query_vi → response_vi`, chuẩn hóa dòng đáp án cuối về `Đáp án là: ...`, giữ ký hiệu toán/LaTeX, drop duplicate/conflict nhỏ, và kiểm soát mẫu quá dài để không mất đáp án khi truncate. Cách này phù hợp với GPT-2 causal LM và bám sát scoring vì model sẽ học nhất quán cách sinh lời giải + đáp án cuối dễ extract.
