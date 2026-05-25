# Tạo variant cho bài toán Fine-tune GPT-2 cho toán tiếng việt

Dưới đây là cách thiết kế **4 variant độc lập từ một baseline**. Mục tiêu là mỗi variant chỉ thay **một cơ chế chính**, để sau khi chạy bạn biết cải thiện đến từ đâu, chứ không bị lẫn giữa data, decoding, verifier và curriculum.

Trước khi chạy 4 variant, bạn nên giữ cố định các yếu tố sau cho mọi thí nghiệm: cùng validation set, cùng metric, cùng answer extractor, cùng seed nếu có thể, cùng base model/checkpoint ban đầu, cùng số mẫu evaluate. Nếu không giữ cố định, rất khó kết luận variant nào thật sự tốt hơn baseline.

# Baseline gốc cần được hiểu là gì?

Baseline hiện tại của bạn có đặc điểm:

```
Model: GPT-2 + LoRA 
Training: train trên dữ liệu đã clean cơ bản 
Inference: pure model 
Decoding: beam2, do_sample=false 
Không retrieval 
Không rule 
Không voting
```

Report trước cho thấy:

```
extractable_rate ≈ 0.999 
anchor_found_rate ≈ 0.999 
loop_rate = 0 
arithmetic_fail_rate_checked ≈ 0.978
```

Nghĩa là baseline **không yếu chủ yếu ở format**, mà yếu ở **tính toán, lập luận và chọn đáp án**. Dữ liệu của bạn cũng là bài toán `query_vi → response_vi`, trong đó `type` nên dùng để phân tích lỗi/stratified evaluation hơn là coi như input bắt buộc của bài toán. data understanding

# Variant 1 — Chuẩn hóa lời giải trung gian mức B bằng script auto-format

## 1. Variant này là làm gì?

Variant này thay đổi **dữ liệu target khi fine-tune**.

Baseline dùng `response_vi` gần như theo format gốc. Variant 1 sẽ tạo một phiên bản mới của `response_vi`, gọi là ví dụ:

```
response_vi_stepfmt
```

Trong đó lời giải được format lại thành:

```
Lời giải: 
Bước 1: ... 
Bước 2: ... 
Bước 3: ... 
... 
Đáp án là: ...
```

Nhưng đây là **chuẩn hóa mức B**, nghĩa là:

```
không rewrite lại nội dung toán học 
không tự bịa thêm bước 
không ép mọi bài thành đúng 3 bước 
chỉ chia lời giải gốc thành các bước ngắn hơn, nhất quán hơn
```

Lý do có cơ sở: nghiên cứu về scratchpad cho thấy language model làm tốt hơn ở multi-step computation khi được yêu cầu sinh các bước trung gian thay vì chỉ trả lời trực tiếp; tuy nhiên các bước trung gian phải giúp biểu diễn quá trình tính toán, không chỉ là trang trí format. [arXiv+1](https://arxiv.org/abs/2112.00114?utm_source=chatgpt.com)

## 2. Mục tiêu của variant này

Variant này nhằm kiểm tra giả thuyết:

> Nếu target lời giải có format từng bước ổn định hơn, GPT-2 sẽ học cách sinh lời giải có cấu trúc hơn, giảm output lan man, giảm lỗi format nội bộ và có thể cải thiện answer accuracy.

Mục tiêu trực tiếp không phải là “làm model biết toán ngay”, mà là:

```
- output ngắn hơn và có cấu trúc hơn 
- lời giải ít lan man hơn 
- model ít sinh nhiều số vô nghĩa hơn 
- final answer vẫn extract được 
- có thể giảm arithmetic_inconsistent nếu các bước được tách rõ hơn
```

## 3. Vì sao cần làm variant này?

Hiện tại `response_vi` có nhiều kiểu khác nhau:

```
Đáp án là: 
Câu trả lời là: 
#### 
\boxed{...} 
văn xuôi dài 
LaTeX 
một số lỗi dịch như \đóng hộp
```

Nếu train GPT-2 trên nhiều format như vậy, model học một phân phối output lẫn lộn. GPT-2 là causal LM, nó không “hiểu instruction” mạnh như instruction model; nó học cách sinh chuỗi token giống dữ liệu train. Vì vậy nếu muốn nó sinh lời giải từng bước ổn định, target train cũng nên có format ổn định.

## 4. Làm điều đó ra sao?

Quy trình variant 1 nên là:

```
Bước 1: lấy baseline data đã clean 
Bước 2: với mỗi response_vi, tách phần lời giải và phần đáp án cuối 
Bước 3: chuẩn hóa anchor cuối về "Đáp án là: " 
Bước 4: tách phần lời giải thành các câu hoặc cụm lập luận ngắn 
Bước 5: đánh số thành Bước 1, Bước 2, ... 
Bước 6: ghép lại thành response_vi_stepfmt 
Bước 7: fine-tune lại GPT-2 trên query_vi → response_vi_stepfmt 
Bước 8: evaluate bằng cùng valid set và cùng extractor
```

Điểm quan trọng: không được để script “sáng tác lại lời giải”. Script chỉ nên format lại nội dung đã có.

Ví dụ tư duy:

```
Response gốc: 
Tom có 10 đĩa tạ, mỗi đĩa 30 pound. Tổng trọng lượng là 10 * 30 = 300 pound. Khi hạ xuống nặng hơn 20%, nên trọng lượng là 300 * 1.2 = 360 pound. Đáp án là: 360 
 
Sau format: 
Lời giải: 
Bước 1: Tom có 10 đĩa tạ, mỗi đĩa 30 pound. 
Bước 2: Tổng trọng lượng là 10 * 30 = 300 pound. 
Bước 3: Khi hạ xuống nặng hơn 20%, nên trọng lượng là 300 * 1.2 = 360 pound. 
Đáp án là: 360
```

## 5. Nên áp dụng cho toàn bộ dữ liệu hay một phần?

Để variant rõ ràng, bạn có thể áp dụng cho **toàn bộ train**. Nhưng nếu muốn an toàn hơn, nên có hai bước nội bộ:

```
1. Áp dụng chắc chắn cho GSM 
2. Với MATH quá dài, LaTeX phức tạp hoặc [asy], chỉ chuẩn hóa đáp án cuối, không tách quá mạnh
```

Tuy nhiên, vì mục tiêu là tạo **một variant độc lập**, có thể đặt tên rõ:

```
variant_1_step_format_all_safe
```

Trong đó “safe” nghĩa là: nếu response quá dài, nhiều LaTeX hoặc không tách được câu ổn định, giữ nguyên lời giải nhưng vẫn chuẩn hóa final answer.

## 6. Những gì không được làm trong variant này?

Không dùng self-consistency.
Không dùng rule verifier.
Không đổi curriculum.
Không đổi train order GSM → MATH.
Không đổi decoding ngoài mức cần thiết để evaluate giống baseline.

Nếu đổi nhiều thứ cùng lúc, bạn sẽ không biết kết quả đến từ step-format hay từ decoding.

## 7. Metric cần so sánh với baseline

So sánh:

```
overall score_10 
bucket_10 rate 
bucket_0 rate 
score_10_by_type 
arithmetic_inconsistent count 
wrong_numeric_answer count 
average output length 
extractable_rate 
anchor_found_rate
```

Variant này được xem là thành công nếu:

```
score_10 tăng 
bucket_0 giảm 
GSM_Rephrased/GSM_AnsAug tăng 
arithmetic_inconsistent giảm hoặc không tăng 
extractable_rate vẫn gần 1.0 
output không dài bất thường
```

Nếu score không tăng nhưng output ổn định hơn, variant này vẫn có thể hữu ích để kết hợp sau này, nhưng trong thí nghiệm độc lập thì kết luận là “format hóa chưa đủ để tăng accuracy”.

# Variant 2 — Self-consistency mức 1: majority vote

## 1. Variant này là làm gì?

Variant này **không thay đổi training data** và **không train lại model**.

Nó chỉ thay đổi **cách inference/decoding**.

Baseline hiện tại sinh một output bằng beam search. Variant 2 sẽ sinh nhiều output khác nhau cho cùng một câu hỏi, extract đáp án cuối từ từng output, rồi chọn đáp án xuất hiện nhiều nhất.

Công thức:

```
1 câu hỏi 
→ sinh N lời giải 
→ extract N đáp án 
→ normalize đáp án 
→ majority vote 
→ chọn đáp án cuối
```

Self-consistency được đề xuất như một decoding strategy thay cho greedy decoding: sample nhiều reasoning paths rồi chọn đáp án nhất quán nhất. Paper gốc báo cáo cải thiện trên nhiều benchmark reasoning, trong đó có GSM8K. [arXiv+1](https://arxiv.org/abs/2203.11171?utm_source=chatgpt.com)

## 2. Mục tiêu của variant này

Variant này kiểm tra giả thuyết:

> Baseline có thể đôi khi sinh đúng, nhưng beam search một lần chọn nhầm. Nếu sinh nhiều candidate và vote đáp án, score sẽ tăng.

Mục tiêu là tăng:

```
bucket_10 
score_10
```

mà **không cần train lại**.

## 3. Vì sao có thể hiệu quả?

Toán word problem thường có một đáp án duy nhất nhưng có nhiều đường lập luận khác nhau. Nếu model có xác suất sinh đúng đáp án ở một số lần sampling, majority vote có thể kéo đáp án đúng lên. Self-consistency tận dụng ý tưởng này bằng cách marginalize qua nhiều reasoning paths thay vì tin vào một path duy nhất. [arXiv](https://arxiv.org/abs/2203.11171?utm_source=chatgpt.com)

Với GPT-2, hiệu quả sẽ không mạnh như LLM lớn, nhưng đây là variant rẻ về training vì chỉ tốn inference.

## 4. Làm điều đó ra sao?

Quy trình variant 2:

```
Bước 1: dùng đúng checkpoint baseline 
Bước 2: đổi decoding từ beam deterministic sang sampling 
Bước 3: với mỗi query, sinh N candidate 
Bước 4: extract final answer từ từng candidate 
Bước 5: normalize answer 
Bước 6: đếm tần suất từng answer 
Bước 7: chọn answer có tần suất cao nhất 
Bước 8: nếu hòa, dùng tie-break rule đơn giản 
Bước 9: chấm bằng cùng metric baseline
```

Nên thử N theo các mức:

```
N = 4 
N = 8 
N = 16
```

Nếu thời gian inference hạn chế, bắt đầu với `N=8`.

Sampling nên khác baseline:

```
do_sample = true 
temperature > 0 
top_p hoặc top_k 
num_beams = 1
```

Không nên dùng beam trong self-consistency mức 1, vì bạn cần đa dạng candidate.

## 5. Tie-break nên làm thế nào?

Vì đây là mức 1, tie-break phải đơn giản, không dùng verifier phức tạp.

Có thể chọn theo thứ tự:

```
1. đáp án xuất hiện nhiều nhất 
2. nếu hòa, chọn đáp án từ candidate có logprob tốt hơn nếu có 
3. nếu không có logprob, chọn candidate ngắn hơn và có anchor rõ 
4. nếu vẫn hòa, chọn candidate đầu tiên trong nhóm hòa
```

Không dùng arithmetic checker ở variant này. Nếu dùng checker, nó sẽ biến thành variant 3.

## 6. Những gì không được làm trong variant này?

Không train lại.
Không thay data.
Không rule-check phép tính.
Không loại candidate vì sai arithmetic.
Không curriculum.

Variant 2 phải thuần majority vote để bạn đo riêng tác động của self-consistency.

## 7. Metric cần so sánh với baseline

So sánh:

```
score_10 
bucket_10 
bucket_0 
score_10_by_type 
extractable_rate 
parse_or_no_number 
inference_time_per_1000_samples
```

Variant này thành công nếu:

```
score_10 tăng rõ so với baseline 
bucket_10 tăng 
bucket_0 giảm 
parse_or_no_number không tăng nhiều
```

Nếu score tăng nhưng inference time quá lớn, bạn cần cân nhắc N nhỏ hơn.

## 8. Kết luận mong đợi

Variant 2 sẽ cho bạn biết:

```
model có "biết đáp án đúng một phần" nhưng baseline decode chọn sai không?
```

Nếu majority vote không cải thiện, nghĩa là model có thể sai có hệ thống; khi đó cần variant 3 hoặc train lại.

# Variant 3 — Self-consistency mức 2: rule-verifier selection

## 1. Variant này là làm gì?

Variant này cũng **không train lại model**, nhưng thay đổi cách chọn đáp án từ nhiều candidate.

Khác variant 2:

```
Variant 2: chọn đáp án bằng majority vote 
Variant 3: chọn candidate bằng rule-verifier score
```

Tức là variant 3 không lấy “đáp án xuất hiện nhiều nhất” làm tiêu chí chính. Nó sinh nhiều candidate rồi chấm từng candidate bằng một bộ rule, sau đó chọn candidate có điểm verifier cao nhất.

OpenAI từng đề xuất hướng generate nhiều candidate rồi dùng verifier để chọn lời giải đúng nhất cho GSM8K; paper của họ cho thấy verification cải thiện performance so với chỉ fine-tuning/generation. [arXiv+1](https://arxiv.org/abs/2110.14168?utm_source=chatgpt.com) Variant của bạn là bản rule-based, không train verifier.

## 2. Mục tiêu của variant này

Variant này kiểm tra giả thuyết:

> Model sinh nhiều candidate, nhưng đa số có lỗi số học rõ ràng. Nếu loại/phạt candidate có lỗi arithmetic hoặc format bất thường, ta chọn được đáp án tốt hơn majority vote.

Mục tiêu là đánh thẳng vào lỗi lớn nhất trong report:

```
arithmetic_fail_rate_checked ≈ 0.978
```

## 3. Vì sao variant này cần tách biệt với variant 2?

Nếu bạn làm:

```
vote trước 
rồi rule sau
```

thì khó biết cải thiện đến từ voting hay rule.

Vì vậy variant 3 nên độc lập:

```
cùng checkpoint baseline 
cùng số candidate N với variant 2 
nhưng selection khác: 
- không chọn theo tần suất đáp án 
- chọn theo rule-verifier score
```

Như vậy khi so sánh variant 2 và 3, bạn biết rule-verifier có giá trị thêm không.

## 4. Làm điều đó ra sao?

Quy trình variant 3:

```
Bước 1: dùng checkpoint baseline 
Bước 2: sinh N candidate bằng sampling, giống variant 2 
Bước 3: extract answer từng candidate 
Bước 4: normalize answer 
Bước 5: tính rule-verifier score cho từng candidate 
Bước 6: chọn candidate có verifier score cao nhất 
Bước 7: lấy final answer của candidate đó 
Bước 8: chấm bằng cùng metric baseline
```

## 5. Rule-verifier nên chấm cái gì?

Vì bạn không viết learned verifier, rule nên đơn giản và trực tiếp.

Nhóm rule nên có:

### Rule A — Format hợp lệ

Candidate được cộng điểm nếu:

```
có anchor "Đáp án là:" 
extract được đáp án 
đáp án không rỗng 
đáp án không quá dài bất thường
```

Bị phạt nếu:

```
không có answer 
có nhiều anchor mâu thuẫn 
đáp án chứa chuỗi số vô nghĩa quá dài
```

### Rule B — Arithmetic consistency

Tìm các biểu thức rõ như:

```
a + b = c 
a - b = c 
a * b = c 
a / b = c
```

Nếu phép tính đúng, cộng điểm.
Nếu phép tính sai, phạt mạnh.

Rule này phù hợp vì report của bạn có nhiều lỗi kiểu:

```
10 * 30 = 60 
5 * 5 = 30 
105 + 105 = 108
```

### Rule C — Không copy số trong đề quá lộ

Nếu đáp án cuối trùng nguyên một số trong query nhưng lời giải không chứng minh được, phạt nhẹ.

Không nên phạt quá mạnh, vì nhiều bài đáp án đúng có thể là một số xuất hiện trong đề.

### Rule D — Output length sanity

Phạt nếu output:

```
quá ngắn, chỉ đoán đáp án 
quá dài bất thường 
có chuỗi lặp số 
có biểu thức LaTeX bị vỡ nặng
```

### Rule E — Bám query numbers

Candidate được cộng nhẹ nếu lời giải dùng các số chính trong đề.
Bị phạt nếu sinh nhiều số lạ không liên quan.

## 6. Cách chọn candidate

Không nên chọn theo answer frequency. Nên chọn theo:

```
verifier_score cao nhất
```

Nếu hòa:

```
1. candidate có nhiều phép tính đúng hơn 
2. candidate có ít phép tính sai hơn 
3. candidate ngắn hơn 
4. candidate có answer xuất hiện nhiều hơn trong các candidate khác
```

Điểm cuối cùng có thể dùng majority vote làm tie-break, nhưng không phải tiêu chí chính.

## 7. Những gì không được làm trong variant này?

Không train lại model.
Không đổi dữ liệu.
Không dùng learned verifier.
Không dùng majority vote làm tiêu chí chính.
Không curriculum.

## 8. Metric cần so sánh

So sánh với baseline và variant 2:

```
score_10 
bucket_10 
bucket_0 
arithmetic_inconsistent 
arithmetic_fail_rate_checked 
wrong_numeric_answer 
copy_input_number 
parse_or_no_number 
inference_time
```

Variant 3 thành công nếu:

```
score_10 > baseline 
score_10 >= variant 2 
arithmetic_inconsistent giảm rõ 
arithmetic_fail_rate_checked giảm 
parse_or_no_number không tăng
```

Nếu variant 3 giảm arithmetic error nhưng score không tăng, có thể rule quá bảo thủ hoặc chọn candidate “ít sai phép tính” nhưng không đúng đáp án.

## 9. Kết luận mong đợi

Variant 3 trả lời câu hỏi:

```
Có thể tăng score bằng cách chọn candidate tốt hơn mà không cần train lại không?
```

Nếu variant 3 thắng variant 2, rule-verifier đáng giữ. Nếu variant 2 thắng variant 3, rule của bạn có thể đang phạt nhầm hoặc không bắt đúng lỗi.

# Variant 4 — Curriculum train theo GSM → MATH

## 1. Variant này là làm gì?

Variant này thay đổi **quá trình training**.

Baseline train trên dữ liệu trộn tương đối đều hoặc theo pipeline hiện tại. Variant 4 sẽ train theo thứ tự từ nhóm dễ/nền tảng sang nhóm khó/phức tạp:

```
Stage 1: GSM_AnsAug + GSM_Rephrased 
Stage 2: thêm GSM_SV + GSM_FOBAR 
Stage 3: thêm MATH_AnsAug + MATH_Rephrased 
Stage 4: thêm MATH_SV + MATH_FOBAR
```

Đây là curriculum learning: học từ dễ/ổn định trước rồi dần thêm nhóm khó hơn. Curriculum learning được đề xuất như một cách giúp quá trình tối ưu dễ hơn và có thể cải thiện generalization trong một số bối cảnh. [ACM Digital Library+1](https://dl.acm.org/doi/10.1145/1553374.1553380?utm_source=chatgpt.com)

## 2. Mục tiêu của variant này

Variant này kiểm tra giả thuyết:

> GPT-2 chưa có nền arithmetic tốt; nếu học GSM direct trước, sau đó mới học biến thể SV/FOBAR và MATH, model sẽ ổn định hơn so với train toàn bộ hỗn hợp ngay từ đầu.

Mục tiêu:

```
- tăng GSM_Rephrased 
- tăng GSM_AnsAug 
- giảm arithmetic_inconsistent 
- sau đó giữ được GSM khi thêm SV/FOBAR và MATH 
- tăng overall score_10
```

## 3. Vì sao curriculum này hợp lý?

GSM8K gồm các bài toán cần 2–8 bước với phép tính cơ bản `+ - * /`, nên nó phù hợp làm nền trước cho GPT-2. [GitHub](https://github.com/openai/grade-school-math?utm_source=chatgpt.com) Trong khi đó dữ liệu của bạn có cả GSM, MATH, SV, FOBAR, LaTeX, symbolic answer và nhiều augmentation; nếu trộn tất cả từ đầu, model nhỏ như GPT-2 dễ học pattern bề mặt thay vì quy trình giải. data understanding

Ngoài ra report trước của bạn cho thấy `GSM_Rephrased` và `GSM_AnsAug` là hai nhóm thấp nhất, nên ưu tiên chúng trước là quyết định dựa trên metric chứ không phải cảm tính.

## 4. Làm điều đó ra sao?

Variant 4 nên dùng cùng data format với baseline, tức **không áp dụng step-format của variant 1**, để giữ độc lập.

Quy trình:

```
Bước 1: bắt đầu từ cùng base model/checkpoint như baseline 
Bước 2: train stage 1 trên GSM direct 
Bước 3: evaluate checkpoint stage 1 
Bước 4: train tiếp stage 2 với mix GSM direct + GSM SV/FOBAR 
Bước 5: evaluate checkpoint stage 2 
Bước 6: train tiếp stage 3 với mix GSM + MATH direct 
Bước 7: evaluate checkpoint stage 3 
Bước 8: train tiếp stage 4 với mix tất cả type 
Bước 9: chọn checkpoint tốt nhất theo validation metric
```

Điểm quan trọng: khi thêm stage mới, **không bỏ hoàn toàn stage cũ**, vì model có thể quên GSM.

## 5. Sampling cụ thể theo stage

### Stage 1 — GSM direct

Dữ liệu:

```
GSM_Rephrased 
GSM_AnsAug
```

Sampling đề xuất:

```
50% GSM_Rephrased 
50% GSM_AnsAug
```

Hoặc nếu muốn nhấn nhóm yếu nhất:

```
60% GSM_Rephrased 
40% GSM_AnsAug
```

Mục tiêu stage này:

```
GSM direct score tăng 
arithmetic_inconsistent giảm 
output vẫn extract được
```

### Stage 2 — thêm GSM biến đổi

Dữ liệu:

```
GSM_Rephrased 
GSM_AnsAug 
GSM_SV 
GSM_FOBAR
```

Sampling đề xuất:

```
35% GSM_Rephrased 
35% GSM_AnsAug 
15% GSM_SV 
15% GSM_FOBAR
```

Lý do: SV/FOBAR là task khác direct-answer, nhưng không nên để chúng chiếm quá nhiều ngay, nếu không model có thể lẫn giữa “trả đáp án trực tiếp” và “tìm biến x / suy ngược tham số”.

Mục tiêu stage này:

```
GSM_SV tăng 
GSM_FOBAR tăng 
GSM direct không giảm mạnh 
wrong_solve_for_variable giảm 
wrong_reverse_param giảm
```

### Stage 3 — thêm MATH direct

Dữ liệu:

```
GSM direct 
GSM SV/FOBAR 
MATH_AnsAug 
MATH_Rephrased
```

Sampling đề xuất:

```
25% GSM_Rephrased 
25% GSM_AnsAug 
10% GSM_SV 
10% GSM_FOBAR 
15% MATH_AnsAug 
15% MATH_Rephrased
```

Lý do: thêm MATH nhưng vẫn giữ GSM làm neo để tránh quên arithmetic.

Mục tiêu stage này:

```
MATH_AnsAug tăng 
MATH_Rephrased tăng 
GSM không tụt nhiều 
copy_input_number không tăng
```

### Stage 4 — thêm MATH còn lại

Dữ liệu:

```
tất cả 8 type
```

Sampling đề xuất:

```
20% GSM_Rephrased 
20% GSM_AnsAug 
10% GSM_SV 
10% GSM_FOBAR 
15% MATH_AnsAug 
15% MATH_Rephrased 
5% MATH_SV 
5% MATH_FOBAR
```

Lý do: `MATH_SV` và `MATH_FOBAR` có n nhỏ và khó hơn, nên không nên để chúng chiếm quá lớn.

Mục tiêu stage này:

```
overall score tăng 
MATH_SV/MATH_FOBAR không quá tệ 
GSM giữ được
```

## 6. Checkpoint nên chọn thế nào?

Không nhất thiết checkpoint cuối stage 4 là tốt nhất.

Bạn nên lưu:

```
checkpoint_stage1 
checkpoint_stage2 
checkpoint_stage3 
checkpoint_stage4
```

Sau đó chọn theo:

```
primary: overall score_10 
secondary: GSM average score 
secondary: bucket_10 
secondary: bucket_0 
safety: extractable_rate >= baseline
```

Nếu mục tiêu của bạn là score chung, chọn checkpoint có overall cao nhất. Nếu score chung tăng ít nhưng GSM tăng mạnh, vẫn ghi nhận là curriculum có tác dụng ở GSM.

## 7. Những gì không được làm trong variant này?

Không dùng step-format của variant 1.
Không dùng self-consistency của variant 2.
Không dùng rule-verifier của variant 3.
Không chia MATH theo topic.
Không đổi prompt nếu baseline không đổi prompt.

Variant 4 phải chỉ kiểm tra tác động của train order/sampling.

## 8. Metric cần so sánh

Bắt buộc có:

```
overall score_10 
score_10_by_type 
GSM average score 
MATH average score 
bucket_10_by_type 
bucket_0_by_type 
arithmetic_inconsistent 
wrong_numeric_answer 
copy_input_number 
wrong_solve_for_variable 
wrong_reverse_param 
extractable_rate 
loop_rate
```

Variant 4 thành công nếu:

```
GSM_Rephrased/GSM_AnsAug tăng rõ 
overall score_10 tăng 
bucket_0 giảm 
arithmetic_inconsistent giảm 
MATH không làm GSM tụt quá mạnh khi thêm vào stage sau
```

Nếu stage 1 hoặc 2 tốt hơn stage 4, nghĩa là thêm MATH làm model quên hoặc nhiễu. Khi đó curriculum đúng hướng, nhưng stage mix cần điều chỉnh.

# Cách đặt tên 4 variant để quản lý kết quả

Bạn nên đặt tên rõ như sau:

```
baseline_v0 
variant_1_stepfmt_train 
variant_2_self_consistency_vote 
variant_3_rule_verifier_select 
variant_4_curriculum_gsm_to_math
```

Mỗi variant lưu riêng:

```
config.json 
valid_report.json 
valid_eval_rows.jsonl 
prediction file 
checkpoint path
```

# Bảng phân biệt 4 variant

| Variant | Thay đổi training? | Thay đổi inference? | Mục tiêu chính | Không được lẫn với |
| --- | --- | --- | --- | --- |
| Variant 1: step-format | Có | Không | Làm target lời giải nhất quán hơn | self-consistency, curriculum |
| Variant 2: majority vote | Không | Có | Tận dụng nhiều candidate để chọn đáp án phổ biến | rule-verifier |
| Variant 3: rule-verifier select | Không | Có | Chọn candidate ít lỗi arithmetic/format hơn | majority vote làm tiêu chí chính |
| Variant 4: curriculum | Có | Không | Học GSM trước rồi MATH sau | step-format, voting, verifier |

# Thứ tự chạy khuyến nghị

Nếu có giới hạn thời gian, nên chạy theo thứ tự:

```
1. Variant 2 — nhanh nhất, không train lại 
2. Variant 3 — cũng không train lại, chỉ thêm selection 
3. Variant 1 — cần tạo data mới và train lại 
4. Variant 4 — cần train nhiều stage, tốn nhất
```

Lý do: Variant 2 và 3 cho bạn biết **có thể cải thiện chỉ bằng inference không**. Nếu inference đã giúp tăng rõ, bạn có một hướng rẻ. Variant 1 và 4 kiểm tra **training/data strategy**, tốn thời gian hơn nhưng có thể tạo cải thiện bền hơn.

# Kết luận thực nghiệm

Bạn đang thiết kế đúng: từ một baseline, tạo 4 variant độc lập để kiểm tra 4 giả thuyết khác nhau.

```
Variant 1 kiểm tra: format target từng bước có giúp model học tốt hơn không? 
Variant 2 kiểm tra: model có sinh đúng trong vài sample nhưng beam chọn sai không? 
Variant 3 kiểm tra: rule-verifier có chọn candidate tốt hơn majority vote không? 
Variant 4 kiểm tra: học GSM trước MATH có ổn định và tăng score không?
```

Điểm quan trọng nhất là **không trộn các variant trong lần chạy đầu**. Sau khi biết variant nào hiệu quả, lúc đó mới tính chuyện kết hợp, ví dụ `step-format + curriculum` hoặc `curriculum + rule-verifier`.
