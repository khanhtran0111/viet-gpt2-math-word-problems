# Kế hoạch triển khai Preprocessing
## Fine-tune GPT-2 giải toán tiếng Việt (Math Word Problems)

> **Phạm vi tài liệu:** Tài liệu này mô tả toàn bộ pipeline tiền xử lý dữ liệu từ `train.json` thô đến tập dữ liệu tokenized sẵn sàng huấn luyện. Bao gồm mục tiêu, logic xử lý từng bước, format đầu ra, tiêu chí kiểm tra chất lượng và các lưu ý khi triển khai thực tế.

---

## Mục lục

1. [Tổng quan bài toán và đặc tính dữ liệu](#1-tổng-quan-bài-toán-và-đặc-tính-dữ-liệu)
2. [Mục tiêu preprocessing](#2-mục-tiêu-preprocessing)
3. [Kiến trúc pipeline tổng thể](#3-kiến-trúc-pipeline-tổng-thể)
4. [Bước 1 — Chọn field và sửa artifact rõ ràng](#4-bước-1--chọn-field-và-sửa-artifact-rõ-ràng)
5. [Bước 2 — Clean nhẹ văn bản](#5-bước-2--clean-nhẹ-văn-bản)
6. [Bước 3 — Chuẩn hóa đáp án cuối](#6-bước-3--chuẩn-hóa-đáp-án-cuối)
7. [Bước 4 — Deduplication và xử lý conflict](#7-bước-4--deduplication-và-xử-lý-conflict)
8. [Bước 5 — Xử lý mẫu quá dài](#8-bước-5--xử-lý-mẫu-quá-dài)
9. [Bước 6 — Build prompt và mask loss](#9-bước-6--build-prompt-và-mask-loss)
10. [Bước 7 — Tokenization](#10-bước-7--tokenization)
11. [Format dữ liệu đầu ra](#11-format-dữ-liệu-đầu-ra)
12. [Tiêu chí kiểm tra chất lượng](#12-tiêu-chí-kiểm-tra-chất-lượng)
13. [Lưu ý khi triển khai thực tế](#13-lưu-ý-khi-triển-khai-thực-tế)
14. [Checklist trước khi train](#14-checklist-trước-khi-train)

---

## 1. Tổng quan bài toán và đặc tính dữ liệu

### 1.1 Bài toán

Fine-tune GPT-2 theo dạng **causal language modeling**: model nhận `query_vi` (câu hỏi toán tiếng Việt) và sinh tiếp `response_vi` (lời giải + đáp án). Scoring dựa trên khả năng extract số từ dòng cuối output, so sánh với gold answer theo relative error.

### 1.2 Đặc tính quan trọng cần nắm trước khi xử lý

| Đặc tính | Con số | Tác động đến preprocessing |
|---|---|---|
| Tổng mẫu | 100,000 | Đủ lớn để tolerate drop nhỏ |
| `query_vi` khác `original_question_vi` | 60,619 mẫu | **Không được dùng `original_question_vi` làm input** |
| Exact duplicate cặp (query, response) | 1,032 dòng dư | Drop an toàn |
| Conflict: cùng query, khác answer | 106 query / 447 dòng | Cần resolve, không chỉ drop |
| Anchor đáp án `Đáp án là:` | 74,105 mẫu | Chuẩn anchor duy nhất cần hướng đến |
| Anchor `Câu trả lời là:` | 25,532 mẫu | Cần normalize về anchor duy nhất |
| Mẫu có LaTeX (`\frac`, `\sqrt`...) | 33,685 mẫu | **Không được xóa** |
| Lỗi dịch `\đóng hộp{` | 752 mẫu | Sửa thành `\boxed{` |
| Mẫu có `[asy]` diagram code | 1,012 mẫu | **Luôn strip** — code hình học, GPT-2 không dùng được |
| Decimal dùng dấu phẩy (toàn văn bản) | ~17,173 mẫu | Normalize toàn bộ query + response để nhất quán |
| Không extract được answer | ~203 mẫu | Drop |
| prompt+response > 1,000 chars | 16,871 mẫu (16.87%) | Cần smart truncation |

### 1.3 Phân bố type (giữ để làm stratified validation)

| Type | Số mẫu | Ghi chú |
|---|---|---|
| GSM_AnsAug | 20,329 | Lời văn đời thường, số nguyên |
| GSM_Rephrased | 20,298 | Câu hỏi viết lại |
| MATH_AnsAug | 18,991 | Có LaTeX, biểu thức toán |
| MATH_Rephrased | 12,788 | Tương tự MATH nhưng viết lại |
| GSM_FOBAR | 10,191 | Hỏi ngược, tìm biến ẩn |
| GSM_SV | 9,920 | Hỏi ngược, tìm biến ẩn |
| MATH_FOBAR | 3,769 | FOBAR với bài MATH |
| MATH_SV | 3,714 | SV với bài MATH |

> Field `type` **chỉ lưu vào metadata**, không đưa vào prompt. Dùng để debug và stratified evaluation.

---

## 2. Mục tiêu preprocessing

Preprocessing cho bài toán này có **một mục tiêu cốt lõi**: đảm bảo mọi mẫu train đều có dòng đáp án cuối sạch, nhất quán, dễ extract. Các mục tiêu phụ trợ:

**Phải đạt được:**
- Anchor đáp án cuối đồng nhất: 100% mẫu kết thúc bằng `\nĐáp án là: {answer}`
- Loss không tính trên prompt — chỉ tính trên phần `response_vi`
- Không có exact duplicate hay conflicting target trong tập train
- Ký hiệu toán học và LaTeX được giữ nguyên

**Không nên làm:**
- Không thay `query_vi` bằng `original_question_vi`
- Không ép tất cả đáp án về số nguyên hay số thập phân
- Không xóa LaTeX (`\frac`, `\sqrt`, `$`, `^`, `_`)
- Không truncate mất phần đáp án cuối
- Không đưa `type` vào prompt
- Không normalize decimal chỉ ở dòng đáp án mà bỏ qua phần còn lại — gây mâu thuẫn định dạng trong cùng một sample

---

## 3. Kiến trúc pipeline tổng thể

```
train.json (100,000)
    │
    ▼ Bước 1: Chọn field + sửa artifact
    │   · Giữ query_vi, response_vi, type
    │   · Sửa \đóng hộp{ → \boxed{
    │   · Strip whitespace đầu/cuối
    │
    ▼ Bước 2: Clean nhẹ + strip [asy] + chuẩn hóa decimal toàn văn bản
    │   · Xóa lặp câu biến x
    │   · Xóa English leakage cuối response
    │   · Chuẩn hóa newline thừa
    │   · Strip [asy]...[/asy] khỏi toàn bộ text ← LUÔN LÀM, không phụ thuộc độ dài (có thể
    │     compare có và không có)
    │   · Normalize decimal comma → dấu chấm toàn bộ query + response
    │     (phân biệt decimal "2,5" với phân cách phần nghìn "1.200")
    │
    ▼ Bước 3: Chuẩn hóa đáp án cuối  ←── QUAN TRỌNG NHẤT
    │   · Extract answer từ nhiều pattern
    │   · Normalize số (bỏ unit, giữ LaTeX symbolic)
    │   · Append anchor chuẩn "\nĐáp án là: {answer}"
    │   · Không extract được → DROP (~203 mẫu)
    │
    ▼ Bước 4: Deduplication & Conflict
    │   · Drop 1,032 exact duplicate
    │   · Conflict 447 dòng: MANUAL VALIDATION (106 query — feasible thủ công)
    │   · KHÔNG dedup theo original_question_vi
    │
    ▼ Bước 5: Xử lý mẫu quá dài
    │   · Tokenize thử để đo length (sau khi đã strip [asy])
    │   · Nếu ≤ MAX_LENGTH: giữ nguyên
    │   · Nếu > MAX_LENGTH: smart truncation (cắt giữa, giữ cuối)
    │   · Vẫn > MAX_LENGTH sau truncation: DROP và log
    │
    ▼ Bước 6: Build prompt + mask loss
    │   · Format: "Câu hỏi: {q}\nLời giải:\n{r}"
    │   · labels[prompt_positions] = -100
    │   · labels[pad_positions] = -100
    │
    ▼ Bước 7: Tokenization
    │   · pad_token = eos_token = 50256
    │   · Dynamic padding theo batch
    │
    ▼ Dataset (~98,300 mẫu)
       · Nhất quán anchor "Đáp án là:"
       · Nhất quán decimal format toàn văn bản
       · LaTeX intact, [asy] đã loại bỏ
       · Loss chỉ tính trên response
```

---

## 4. Bước 1 — Chọn field và sửa artifact rõ ràng

### 4.1 Field cần giữ

```python
INPUT_FIELD   = "query_vi"       # Câu hỏi đã augmentation — KHÔNG thay bằng original
TARGET_FIELD  = "response_vi"    # Lời giải + đáp án
METADATA_FIELD = "type"          # Chỉ dùng để phân tích, không đưa vào prompt
```

**Các field bỏ qua hoàn toàn:**
- `original_question_vi` — đây là câu hỏi gốc chưa augmentation, 60,619 mẫu đã bị biến đổi nội dung
- `query_en`, `response_en`, `original_question_en` — không dùng theo yêu cầu bài

### 4.2 Sửa artifact cố định

```python
def fix_artifacts(text: str) -> str:
    # Lỗi dịch \boxed{} — 752 mẫu
    text = text.replace(r"\đóng hộp{", r"\boxed{")

    # Đảm bảo không có BOM hoặc invisible characters
    text = text.replace("\u200b", "").replace("\ufeff", "")

    # Strip whitespace đầu/cuối
    text = text.strip()

    return text
```

> **Lý do phải làm trước:** Artifact `\đóng hộp{` ảnh hưởng đến bước extract answer ở Bước 3. Nếu không sửa trước, pattern matching cho `\boxed{...}` sẽ bỏ sót 752 mẫu.

---

## 5. Bước 2 — Clean nhẹ, strip `[asy]`, chuẩn hóa decimal toàn văn bản

### Nguyên tắc: Clean những gì chắc chắn là nhiễu, giữ nguyên phần còn lại.

### 5.1 Clean văn bản cơ bản

```python
import re

def clean_text(text: str) -> str:
    # 1. Chuẩn hóa nhiều khoảng trắng liên tiếp → 1 khoảng trắng
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # 2. Chuẩn hóa nhiều newline liên tiếp → tối đa 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 3. Xóa lặp câu hỏi biến x nguyên văn (~3,201 mẫu)
    #    Pattern: câu "Giá trị của biến ... là bao nhiêu?" bị lặp 2+ lần
    text = re.sub(
        r'(Giá trị của biến [^\n?]+\?)\s*\1',
        r'\1',
        text
    )

    # 4. Xóa English leakage ở CUỐI response
    #    Chỉ xóa nếu là câu tiếng Anh standalone ở cuối, không xóa trong giữa
    text = re.sub(
        r'\n(?:The answer is[:\s]+[\d.,/\\{}a-zA-Z]+\.?\s*)+$',
        '',
        text,
        flags=re.IGNORECASE
    )

    return text.strip()
```

### 5.2 Strip `[asy]` — luôn làm, không phụ thuộc độ dài / Thử chạy so sánh nhẹ

```python
def strip_asy_blocks(text: str) -> str:
    """
    Xóa toàn bộ block [asy]...[/asy] khỏi văn bản.

    Lý do strip vô điều kiện (không phải chỉ khi quá dài):
    - [asy] là Asymptote code: tọa độ, lệnh draw, path — ngôn ngữ lập trình đồ họa.
    - GPT-2 xử lý text ngôn ngữ tự nhiên, không có khả năng suy luận
      không gian hình học từ tọa độ code thô.
    - Phần mô tả bài toán bằng ngôn ngữ tự nhiên đã chứa đủ thông tin
      toán học cần thiết — [asy] không bổ sung thêm signal nào model có thể dùng.
    - Giữ lại [asy] chỉ tốn token budget và làm phân tán attention vào nhiễu.
    - Tuy nhiên, [asy] vẫn có thể được giữ lại như một biến thử nghiệm riêng để đo xem 
      mô hình có khai thác được thông tin hình học từ diagram code hay không.
    - Nếu test set có [asy]: model vẫn đọc được phần text mô tả bài toán,
      chỉ bỏ qua phần code hình học vô nghĩa — không mất thông tin giải toán.
    """
    # Xóa cả block có hoặc không có newline bao quanh
    text = re.sub(r'\[asy\].*?\[/asy\]', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Dọn dẹp khoảng trắng thừa sau khi xóa block
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()
```

### 5.3 Chuẩn hóa decimal toàn văn bản

**Vấn đề:** Nếu `query_vi` có "2,5 kg" nhưng dòng đáp án cuối bắt buộc là "2.5", model phải học một ánh xạ ngầm comma → dot trong cùng một sample. Với GPT-2 nhỏ, điều này tốn capacity không cần thiết và làm tín hiệu học không ổn định.

**Giải pháp:** Normalize nhất quán trên toàn bộ `query_vi` và `response_vi` — không chỉ dòng đáp án cuối.

**Vấn đề phức tạp:** Tiếng Việt dùng dấu chấm làm phân cách **phần nghìn** ("1.200 đồng" = 1200 đồng), còn dấu phẩy làm **dấu thập phân** ("2,5 kg" = 2.5 kg). Regex phải phân biệt hai trường hợp này.

```python
def normalize_decimal_format(text: str) -> str:
    """
    Chuẩn hóa dấu thập phân từ phẩy sang chấm trên toàn bộ văn bản.

    Quy tắc phân biệt:
    - Decimal (cần đổi):  số,1-2chữsố  →  ví dụ "2,5" "9,75" "-3,14"
    - Phần nghìn (giữ):   số.3chữsố    →  ví dụ "1.200" "10.000"
    - European style:     1.200,5       →  "1200.5" (đổi cả hai)

    Không đổi nếu:
    - Dấu phẩy trong câu văn bình thường: "Hôm nay, trời đẹp"
    - Dấu phẩy trong LaTeX: "\frac{1,2}{3}" → giữ nguyên
    """
    # Bảo vệ LaTeX trước — đánh dấu tạm thời để không bị xử lý
    # Cách đơn giản: xử lý ngoài LaTeX block
    # (LaTeX block được bọc bởi $ hoặc \[ \] hoặc \begin{} \end{})

    def replace_decimal_comma(m):
        """Callback: chỉ đổi nếu là decimal thực sự."""
        before = m.group(1)  # Phần số trước dấu phẩy
        after  = m.group(2)  # Phần số sau dấu phẩy

        # Nếu phần sau có đúng 3 chữ số → khả năng là phân cách phần nghìn
        # Ví dụ: "1,200" trong tiếng Việt thường là 1200, không phải 1.2
        # Nhưng "1,20" hay "1,5" là decimal
        if len(after) == 3 and int(after) % 100 == 0:
            # Heuristic: "X,Y00" với Y00 là bội của 100 → phần nghìn, giữ nguyên
            return m.group(0)

        return f"{before}.{after}"

    # Pattern: số nguyên, dấu phẩy, 1-2 chữ số (decimal rõ ràng)
    # Không match nếu trước/sau là chữ cái (tránh xử lý câu văn)
    text = re.sub(
        r'(?<![a-zA-ZÀ-ỹ])(-?\d+),(\d{1,2})(?!\d)(?![a-zA-ZÀ-ỹ}])',
        replace_decimal_comma,
        text
    )

    # European style: "1.200,5" → "1200.5"
    # Pattern: số.3chữsố,1-2chữsố
    text = re.sub(
        r'(\d+)\.(\d{3}),(\d{1,2})(?!\d)',
        lambda m: f"{m.group(1)}{m.group(2)}.{m.group(3)}",
        text
    )

    return text
```

> **Lưu ý:** Hàm này áp dụng cho **cả `query_vi` lẫn `response_vi`** trước khi ghép prompt. Kết quả là model thấy format số nhất quán trong toàn bộ input và output, không phải "2,5 kg" trong câu hỏi và "2.5" trong đáp án.

### 5.4 Thứ tự gọi trong Bước 2

```python
def preprocess_step2(query: str, response: str) -> tuple[str, str]:
    # 1. Strip [asy] trước — giảm độ dài trước các bước sau
    query    = strip_asy_blocks(query)
    response = strip_asy_blocks(response)

    # 2. Clean text cơ bản
    query    = clean_text(query)
    response = clean_text(response)

    # 3. Normalize decimal — sau clean để không bị ảnh hưởng bởi whitespace cũ
    query    = normalize_decimal_format(query)
    response = normalize_decimal_format(response)

    return query, response
```

### 5.5 Những gì KHÔNG làm trong Bước 2

```
KHÔNG xóa:   \frac{}{}, \sqrt{}, \cdot, \left, \right, ^, _
KHÔNG xóa:   Dấu $ trong bài MATH
KHÔNG xóa:   Số nguyên dùng dấu chấm phân cách phần nghìn: "1.200"
KHÔNG thay:  query_vi bằng original_question_vi
```

---

## 6. Bước 3 — Chuẩn hóa đáp án cuối

### 6.1 Tại sao đây là bước quan trọng nhất

Scoring của bài toán phụ thuộc hoàn toàn vào việc extract đúng con số ở dòng cuối output. Nếu model học nhiều kiểu kết thúc khác nhau (`####`, `Câu trả lời là:`, `\boxed{}`, `Đáp án là:`), khả năng sinh ra đúng format khi inference giảm đáng kể. Mục tiêu là ép toàn bộ tập train về **một anchor duy nhất** ở dòng cuối.

### 6.2 Thứ tự ưu tiên extract

```python
def extract_final_answer(response: str) -> str | None:
    """
    Trả về answer string đã extract, hoặc None nếu không tìm được.
    Thứ tự ưu tiên: anchor tiếng Việt > #### > \boxed{} > số cuối cùng
    """

    # Ưu tiên 1: "Đáp án là: ..." (74,105 mẫu)
    m = re.search(r'Đáp án là[:\s]+(.+?)(?:\s*$)', response, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()

    # Ưu tiên 2: "Câu trả lời là: ..." (25,532 mẫu)
    m = re.search(r'Câu trả lời là[:\s]+(.+?)(?:\s*$)', response, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()

    # Ưu tiên 3: "#### <number>" (GSM style — 59,762 mẫu có pattern này)
    m = re.search(r'####\s*(.+?)(?:\s*$)', response, re.MULTILINE)
    if m:
        return m.group(1).strip()

    # Ưu tiên 4: "\boxed{...}" (29,356 mẫu)
    m = re.search(r'\\boxed\{([^}]+)\}', response)
    if m:
        return m.group(1).strip()

    # Fallback: số cuối cùng trong response
    # Bao gồm: số nguyên, decimal, phân số LaTeX, số âm
    numbers = re.findall(
        r'(?:\\frac\{[^}]+\}\{[^}]+\}|'
        r'-?\d+(?:[.,]\d+)?(?:\s*\\[a-zA-Z]+\{[^}]*\})*)',
        response
    )
    if numbers:
        return numbers[-1].strip()

    return None  # → DROP mẫu này
```

### 6.3 Normalize số ở dòng đáp án cuối

```python
def normalize_answer(answer: str) -> str:
    """
    Normalize DÒNG ĐÁP ÁN CUỐI sau khi đã extract.
    Lúc này decimal đã được chuẩn hóa toàn văn bản ở Bước 2,
    nên chỉ cần xử lý bỏ unit và giữ LaTeX symbolic.
    """
    answer = answer.strip()

    # Bỏ unit đứng sau số ở dòng cuối (giữ unit trong lời giải)
    # Ví dụ: "150 km" → "150", "45 phút" → "45"
    # KHÔNG xóa nếu là LaTeX: "50\sqrt{10}" giữ nguyên
    if not re.search(r'[\\{^_]', answer):
        answer = re.sub(r'^(-?[\d./]+)\s+[a-zA-ZÀ-ỹ]+.*$', r'\1', answer)

    return answer.strip()
```

> **Lưu ý:** Decimal comma đã được chuẩn hóa ở Bước 2 trên toàn văn bản. Tại bước này không cần xử lý lại — chỉ cần bỏ unit và bảo toàn LaTeX symbolic.

### 6.4 Rebuild response với anchor chuẩn

```python
def rebuild_response(response: str, answer: str) -> str:
    """
    Xóa tất cả marker cũ ở cuối response và gắn lại anchor chuẩn duy nhất.
    Giữ toàn bộ phần lời giải ở giữa.
    """
    # Xóa các marker cuối cũ (từ vị trí marker đến hết)
    # Chú ý: xóa cả \boxed{} ở DÒNG CUỐI, nhưng giữ \boxed{} trong lời giải
    cleaned = re.sub(
        r'\n(?:Đáp án là|Câu trả lời là)[^\n]*$',
        '',
        response,
        flags=re.IGNORECASE
    )
    cleaned = re.sub(r'\n####[^\n]*$', '', cleaned)

    # Không xóa \boxed{} trong lời giải — chỉ bỏ nếu là dòng cuối standalone
    cleaned = re.sub(r'\n\\boxed\{[^}]*\}\s*$', '', cleaned)

    return cleaned.rstrip() + f"\nĐáp án là: {answer}"
```

### 6.5 Xử lý mẫu không extract được

```python
# ~203 mẫu không extract được qua cả 4 pattern + fallback
# → DROP và log ra file để kiểm tra thủ công

if extracted_answer is None:
    failed_samples.append({
        "index": idx,
        "query_vi": record["query_vi"][:100],
        "response_tail": record["response_vi"][-200:],
        "type": record["type"]
    })
    continue  # Skip mẫu này
```

---

## 7. Bước 4 — Deduplication và xử lý conflict

### 7.1 Drop exact duplicate

```python
# Tạo key từ cặp (query đã clean, response đã clean)
seen = set()
deduped = []

for record in records:
    key = (record["query_clean"], record["response_clean"])
    if key not in seen:
        seen.add(key)
        deduped.append(record)
    # else: drop — 1,032 dòng dư
```

> **Lý do:** 1,032 exact duplicate không mang thêm thông tin, chỉ tăng nguy cơ overfit trên những mẫu đó.

### 7.2 Resolve conflict (cùng query, khác answer) — Manual Validation

```python
def export_conflicts_for_review(records: list, output_path: str):
    """
    Xuất toàn bộ 106 query conflict ra file CSV/JSON để review thủ công.
    Với quy mô 106 query / 447 dòng, manual validation hoàn toàn feasible
    và cho kết quả đáng tin cậy hơn bất kỳ heuristic tự động nào.
    """
    from collections import defaultdict
    import json

    query_groups = defaultdict(list)
    for r in records:
        query_groups[r["query_clean"]].append(r)

    conflicts = []
    for query, group in query_groups.items():
        unique_answers = list({r["final_answer"] for r in group})
        if len(unique_answers) > 1:
            conflicts.append({
                "query": query,
                "type": group[0]["type"],
                "answer_variants": unique_answers,
                "answer_counts": {
                    a: sum(1 for r in group if r["final_answer"] == a)
                    for a in unique_answers
                },
                "sample_responses": [
                    r["response_clean"][:300] for r in group[:3]
                ],
                # Field để điền khi review: "keep", "drop_all", hoặc answer đúng
                "manual_decision": None
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(conflicts, f, ensure_ascii=False, indent=2)

    print(f"Xuất {len(conflicts)} conflict groups ra {output_path}")
    print("Mở file, điền 'manual_decision' cho từng group, rồi chạy apply_manual_decisions()")
```

```python
def apply_manual_decisions(records: list, decisions_path: str) -> list:
    """
    Đọc kết quả manual validation và áp dụng vào dataset.
    """
    import json

    with open(decisions_path) as f:
        decisions = {d["query"]: d for d in json.load(f)}

    resolved = []
    for r in records:
        query = r["query_clean"]
        if query not in decisions:
            resolved.append(r)  # Không phải conflict → giữ nguyên
            continue

        decision = decisions[query]["manual_decision"]

        if decision == "drop_all":
            continue  # Bỏ toàn bộ nhóm
        elif decision in (None, "pending"):
            # Chưa review → tạm giữ, log cảnh báo
            print(f"CẢNH BÁO: Conflict chưa được review: {query[:60]}...")
            resolved.append(r)
        else:
            # decision là answer cụ thể được chọn
            if r["final_answer"] == decision:
                resolved.append(r)
            # else: drop — đây là dòng có answer sai

    return resolved
```

**Hướng dẫn manual review:**

Khi mở file conflict, mỗi group cần đưa ra một trong 3 quyết định:

| Trường hợp | Quyết định | Ghi vào `manual_decision` |
|---|---|---|
| Một answer rõ ràng đúng | Giữ answer đúng | Ghi số/giá trị answer đúng |
| Đều sai do lỗi augmentation | Bỏ hết | `"drop_all"` |
| Không chắc chắn | Bỏ hết (an toàn hơn) | `"drop_all"` |

> **Lý do dùng manual validation thay vì majority vote:**
> Với chỉ 106 query conflict, đây là con số feasible để xem thủ công trong vài tiếng.
> Majority vote có thể chọn sai nếu lỗi augmentation xảy ra có hệ thống — ví dụ
> một augmentation variant nào đó sinh ra sai answer cho tất cả mẫu trong nhóm,
> khiến answer sai chiếm đa số. Manual review phát hiện được pattern lỗi này;
> majority vote không.

### 7.3 Quy tắc bất biến

```
TUYỆT ĐỐI KHÔNG dedup theo original_question_vi

Lý do: Cùng 1 original_question_vi có thể sinh ra nhiều query_vi khác nhau
với target hoàn toàn khác:
  - Bản gốc: "Cửa hàng bán 150 quả táo..." → đáp án: 150
  - FOBAR:   "Cửa hàng bán x quả táo... Tìm x" → đáp án: 150
  - SV:      "Cửa hàng bán x quả táo... Biết tổng là 450, tìm x" → đáp án: 150
Đây là 3 mẫu train hợp lệ khác nhau, không phải duplicate.
```

---

## 8. Bước 5 — Xử lý mẫu quá dài

### 8.1 Quyết định theo token length (không phải char length)

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
MAX_LENGTH = 512  # Baseline an toàn cho GPT-2

def get_token_length(query: str, response: str) -> int:
    prompt = build_prompt(query, response)  # Xem Bước 6
    return len(tokenizer.encode(prompt))
```

> **Lưu ý:** Đến bước này, `[asy]` đã được strip ở Bước 2 nên các mẫu dài do `[asy]`
> đã được xử lý rồi. Bước này chỉ còn xử lý các mẫu genuinely dài do lời giải nhiều bước.

### 8.2 Chiến lược xử lý theo độ dài

```
token_length ≤ MAX_LENGTH → Giữ nguyên

token_length > MAX_LENGTH → Smart Truncation:
    1. Giữ toàn bộ query_vi (không bao giờ cắt)
    2. Giữ phần cuối response chứa "\nĐáp án là: {answer}"
    3. Cắt bớt phần GIỮA lời giải từ điểm cần thiết
    4. Đảm bảo sau truncation: dòng "Đáp án là:" vẫn là dòng cuối

    Nếu sau smart truncation vẫn > MAX_LENGTH:
        → DROP và log để kiểm tra
```

```python
def smart_truncate(query: str, response: str, max_length: int) -> str | None:
    prompt_prefix = f"Câu hỏi: {query}\nLời giải:\n"
    prefix_tokens = tokenizer.encode(prompt_prefix)

    answer_suffix = "\nĐáp án là: " + extract_final_answer(response)
    suffix_tokens = tokenizer.encode(answer_suffix)

    # Budget còn lại cho phần giữa lời giải
    middle_budget = max_length - len(prefix_tokens) - len(suffix_tokens) - 5  # buffer

    if middle_budget <= 0:
        return None  # Không thể truncate an toàn → DROP

    # Lấy phần giữa lời giải (bỏ dòng anchor cuối)
    middle_text = re.sub(r'\nĐáp án là:.*$', '', response, flags=re.IGNORECASE).strip()
    middle_tokens = tokenizer.encode(middle_text)

    if len(middle_tokens) > middle_budget:
        # Cắt bớt từ đầu phần giữa, giữ phần cuối gần đáp án hơn
        middle_tokens = middle_tokens[-middle_budget:]

    middle_text_truncated = tokenizer.decode(middle_tokens)
    return middle_text_truncated + answer_suffix
```

### 8.3 Ước lượng tác động

Dựa vào thống kê char length và tỷ lệ token/char ≈ 0.45 cho tiếng Việt với GPT-2 tokenizer:

| Ngưỡng | Char | Ước lượng % bị ảnh hưởng |
|---|---|---|
| MAX_LENGTH = 512 | ~1,140 chars | ~16.87% cần truncation |
| MAX_LENGTH = 768 | ~1,710 chars | ~5–7% cần truncation |
| MAX_LENGTH = 1024 | ~2,280 chars | ~2.85% cần truncation |

> **Khuyến nghị:** Dùng `MAX_LENGTH=512` cho baseline đầu tiên. Nếu VRAM cho phép, tăng lên 768 để giảm thông tin bị mất ở nhóm MATH dài.

---

## 9. Bước 6 — Build prompt và mask loss

### 9.1 Template prompt

```python
PROMPT_TEMPLATE = "Câu hỏi: {query}\nLời giải:\n{response}"

def build_prompt(query: str, response: str) -> str:
    return PROMPT_TEMPLATE.format(query=query.strip(), response=response.strip())
```

**Lý do chọn template này:**
- Ngắn, ít token overhead — quan trọng với GPT-2 nhỏ
- Không dùng `### Bài toán: / ### Lời giải:` vì `###` tốn thêm token và không cần thiết
- Dấu `\n` phân tách rõ boundary giữa câu hỏi và lời giải

### 9.2 Mask loss cho phần prompt

```python
def build_labels(input_ids: list[int], query: str) -> list[int]:
    """
    Loss chỉ tính trên phần response_vi.
    Phần prompt và padding đều được mask bằng -100.
    """
    IGNORE_INDEX = -100

    # Tìm chiều dài của phần prompt (không tính response)
    prompt_only = f"Câu hỏi: {query}\nLời giải:\n"
    prompt_token_len = len(tokenizer.encode(prompt_only))

    labels = input_ids.copy()

    # Mask phần prompt
    for i in range(min(prompt_token_len, len(labels))):
        labels[i] = IGNORE_INDEX

    return labels
```

> **Tác động:** Không mask prompt → model lãng phí capacity học cách sinh lại câu hỏi thay vì tập trung vào lời giải. Với GPT-2 nhỏ, điều này ảnh hưởng rõ rệt đến chất lượng.

---

## 10. Bước 7 — Tokenization

### 10.1 Cấu hình tokenizer

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# GPT-2 không có pad token riêng
# Dùng eos_token làm pad_token theo yêu cầu bài
SAFE_EOS_ID = 50256
tokenizer.pad_token    = tokenizer.eos_token   # "<|endoftext|>"
tokenizer.pad_token_id = SAFE_EOS_ID
tokenizer.eos_token_id = SAFE_EOS_ID
```

### 10.2 Data Collator với dynamic padding

```python
from transformers import DataCollatorForLanguageModeling

# Dynamic padding per batch — KHÔNG pad toàn bộ dataset từ trước
# Lý do: tiết kiệm memory, batch hiệu quả hơn, Hugging Face khuyến nghị
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,           # Causal LM, không phải masked LM
    pad_to_multiple_of=8 # Tăng hiệu suất trên GPU với Tensor Cores
)
```

### 10.3 Mask padding trong labels

```python
# Trong collator hoặc custom __getitem__:
# Sau khi padding, mask vị trí padding trong labels

def mask_padding_in_labels(labels: list[int], attention_mask: list[int]) -> list[int]:
    return [
        label if mask == 1 else -100
        for label, mask in zip(labels, attention_mask)
    ]
```

### 10.4 Tóm tắt cấu hình token

| Tham số | Giá trị | Lý do |
|---|---|---|
| `pad_token_id` | 50256 | Dùng EOS làm PAD — yêu cầu bài |
| `eos_token_id` | 50256 | Đồng nhất để model biết điểm dừng |
| `MAX_LENGTH` | 512 | Baseline an toàn, ~83% mẫu không bị truncate |
| `labels[prompt]` | -100 | Không tính loss trên câu hỏi |
| `labels[pad]` | -100 | Không phạt model vì padding |
| Dynamic padding | Theo batch | Tiết kiệm memory, hiệu suất cao hơn |

---

## 11. Format dữ liệu đầu ra

### 11.1 Schema mỗi mẫu sau preprocessing

```python
{
    # Input cho model
    "input_ids":      [int, ...],   # Tokenized prompt+response, max MAX_LENGTH
    "attention_mask": [int, ...],   # 1 = real token, 0 = padding

    # Labels cho loss
    "labels":         [int, ...],   # -100 ở vị trí prompt và padding
                                    # token_id ở vị trí response

    # Metadata (không đưa vào training, dùng để debug)
    "type":           str,          # GSM_AnsAug, MATH_Rephrased, ...
    "final_answer":   str,          # Answer đã extract và normalize
    "original_length": int,         # Token length trước truncation
    "was_truncated":  bool,         # True nếu đã smart truncate
}
```

### 11.2 Thống kê mẫu đầu ra mong đợi

```
Đầu vào:     100,000 mẫu
Drop (extract fail):          ~203 mẫu
Drop (exact duplicate):      ~1,032 mẫu
Drop (conflict — manual):    ~100-400 mẫu (tùy kết quả review thủ công)
Drop (quá dài sau truncation): ~100-300 mẫu
[asy] stripped (không drop):  ~1,012 mẫu (giữ lại phần text còn lại)

Ước lượng đầu ra:    ~98,100 - 98,600 mẫu
```

### 11.3 Ví dụ mẫu sau preprocessing

**Input (`query_vi`):**
```
Giá trị của biến x chưa biết trong bài toán sau là bao nhiêu?
Một cửa hàng bán được x kg gạo trong ngày đầu. Ngày thứ hai bán gấp đôi.
Tổng hai ngày bán được 450 kg. Tìm x.
```

**Target (`response_vi` sau rebuild):**
```
Gọi số kg gạo bán ngày đầu là x.
Ngày thứ hai bán được: 2x kg.
Tổng hai ngày: x + 2x = 3x = 450.
Suy ra: x = 450 ÷ 3 = 150.
Đáp án là: 150
```

**Prompt hoàn chỉnh đưa vào model:**
```
Câu hỏi: Giá trị của biến x chưa biết trong bài toán sau là bao nhiêu? [...]
Lời giải:
Gọi số kg gạo bán ngày đầu là x. [...]
Đáp án là: 150
```

---

## 12. Tiêu chí kiểm tra chất lượng

Sau khi chạy pipeline, cần verify đủ 5 nhóm tiêu chí sau trước khi bắt đầu train:

### 12.1 Tính nhất quán của anchor đáp án

```python
def check_anchor_consistency(dataset):
    total = len(dataset)
    has_anchor = sum(1 for r in dataset if "Đáp án là:" in r["response_clean"])
    no_anchor = total - has_anchor

    print(f"Tổng mẫu: {total}")
    print(f"Có anchor 'Đáp án là:': {has_anchor} ({has_anchor/total*100:.2f}%)")
    print(f"Không có anchor: {no_anchor}")  # Phải = 0

    # Kiểm tra anchor luôn là dòng cuối
    not_last_line = sum(
        1 for r in dataset
        if not r["response_clean"].rstrip().endswith(("Đáp án là: " + r["final_answer"]))
    )
    print(f"Anchor không phải dòng cuối: {not_last_line}")  # Phải = 0
```

**Tiêu chí pass:**
- 100% mẫu có anchor `Đáp án là:` → `no_anchor = 0`
- 100% anchor là dòng cuối → `not_last_line = 0`
- 0% còn anchor `Câu trả lời là:` trong dataset

### 12.2 Tính sạch của duplicate và conflict

```python
def check_deduplication(dataset):
    keys = [(r["query_clean"], r["response_clean"]) for r in dataset]
    n_unique = len(set(keys))
    n_dup = len(keys) - n_unique
    print(f"Exact duplicate còn lại: {n_dup}")  # Phải = 0

    from collections import Counter
    query_answer_map = {}
    n_conflict = 0
    for r in dataset:
        q = r["query_clean"]
        a = r["final_answer"]
        if q in query_answer_map:
            if query_answer_map[q] != a:
                n_conflict += 1
        else:
            query_answer_map[q] = a
    print(f"Query conflict còn lại: {n_conflict}")  # Nên = 0 hoặc rất nhỏ
```

### 12.3 Bảo toàn LaTeX và loại bỏ `[asy]`

```python
def check_latex_and_asy(dataset_before, dataset_after):
    # LaTeX preservation
    before_latex = sum(1 for r in dataset_before if '\\' in r["response_vi"])
    after_latex  = sum(1 for r in dataset_after  if '\\' in r["response_clean"])

    ratio = after_latex / before_latex
    print(f"LaTeX preservation ratio: {ratio:.3f}")  # Phải > 0.99

    # Kiểm tra không có \đóng hộp{ còn sót
    dongHop = sum(1 for r in dataset_after if r'\đóng hộp{' in r["response_clean"])
    print(f"Còn \\đóng hộp{{: {dongHop}")  # Phải = 0

    # Kiểm tra [asy] đã được strip hoàn toàn
    asy_remaining_q = sum(1 for r in dataset_after if '[asy]' in r.get("query_clean", "").lower())
    asy_remaining_r = sum(1 for r in dataset_after if '[asy]' in r.get("response_clean", "").lower())
    print(f"[asy] còn trong query: {asy_remaining_q}")    # Phải = 0
    print(f"[asy] còn trong response: {asy_remaining_r}") # Phải = 0
```

### 12.4 Tính nhất quán của decimal format

```python
def check_decimal_consistency(dataset):
    """
    Kiểm tra không còn decimal dùng dấu phẩy trong cả query lẫn response,
    đặc biệt là ở dòng đáp án cuối.
    """
    import re

    # Pattern: số nguyên + dấu phẩy + 1-2 chữ số (dạng decimal tiếng Việt)
    decimal_comma_pattern = re.compile(r'(?<!\d)\d+,\d{1,2}(?!\d)')

    query_issues    = sum(1 for r in dataset if decimal_comma_pattern.search(r["query_clean"]))
    response_issues = sum(1 for r in dataset if decimal_comma_pattern.search(r["response_clean"]))
    answer_issues   = sum(1 for r in dataset if decimal_comma_pattern.search(r["final_answer"]))

    print(f"Decimal phẩy còn trong query:    {query_issues}")
    print(f"Decimal phẩy còn trong response: {response_issues}")
    print(f"Decimal phẩy còn trong answer:   {answer_issues}")  # Phải = 0
    print(f"Tổng sample có vấn đề:           {query_issues + response_issues}")
    # Một số nhỏ trong query/response có thể chấp nhận được nếu là edge case
    # nhưng answer_issues phải = 0 tuyệt đối
```

### 12.4 Token length distribution

```python
def check_token_distribution(dataset):
    lengths = [r["original_length"] for r in dataset]

    print(f"Min: {min(lengths)}")
    print(f"Median: {sorted(lengths)[len(lengths)//2]}")
    print(f"P90: {sorted(lengths)[int(len(lengths)*0.9)]}")
    print(f"P99: {sorted(lengths)[int(len(lengths)*0.99)]}")
    print(f"Max: {max(lengths)}")

    over_limit = sum(1 for l in lengths if l > MAX_LENGTH)
    print(f"Vượt MAX_LENGTH ({MAX_LENGTH}): {over_limit}")  # Phải = 0 sau truncation/drop

    # Phân phối theo type
    from collections import defaultdict
    type_lengths = defaultdict(list)
    for r in dataset:
        type_lengths[r["type"]].append(r["original_length"])

    for t, ls in sorted(type_lengths.items()):
        print(f"  {t}: median={sorted(ls)[len(ls)//2]}, max={max(ls)}")
```

### 12.5 Kiểm tra phân phối type sau preprocessing

```python
def check_type_distribution(dataset):
    from collections import Counter
    type_counts = Counter(r["type"] for r in dataset)
    total = len(dataset)

    for t, c in sorted(type_counts.items()):
        print(f"{t}: {c} ({c/total*100:.1f}%)")

    # Cảnh báo nếu nhóm nào bị mất > 5% so với ban đầu
    original = {
        "GSM_AnsAug": 20329, "GSM_Rephrased": 20298,
        "MATH_AnsAug": 18991, "MATH_Rephrased": 12788,
        # ... etc
    }
    for t, orig_count in original.items():
        remaining = type_counts.get(t, 0)
        drop_rate = (orig_count - remaining) / orig_count
        if drop_rate > 0.05:
            print(f"CẢNH BÁO: {t} mất {drop_rate*100:.1f}% mẫu")
```

---

## 13. Lưu ý khi triển khai thực tế

### 13.1 Thứ tự bước không được thay đổi

Các bước phụ thuộc nhau theo thứ tự sau:

```
fix_artifacts()   →   Phải chạy trước extract_answer()
                      vì \đóng hộp{ làm hỏng regex \boxed{}

clean_text()      →   Phải chạy trước rebuild_response()
                      để không duplicate logic clean

extract + rebuild →   Phải chạy trước dedup
                      vì dedup dùng response_clean để so sánh

dedup + conflict  →   Phải chạy trước tokenize
                      để không tokenize mẫu bị drop
```

### 13.2 Logging và traceability

Nên log ra file riêng cho từng loại drop:

```python
logs = {
    "extract_failed":    [],   # ~203 mẫu
    "exact_duplicates":  [],   # ~1,032 mẫu
    "conflict_review":   [],   # File xuất ra để manual review
    "conflict_dropped":  [],   # Sau khi áp dụng manual decisions
    "too_long_dropped":  [],   # Mẫu không truncate được
    "smart_truncated":   [],   # Mẫu đã truncate thành công
    "asy_stripped":      [],   # Mẫu đã strip [asy] (không drop, chỉ log để kiểm tra)
}
```

File log giúp:
- Debug nếu score thấp bất thường ở một nhóm type
- Kiểm tra xem valid/test set có chứa dạng mẫu bị drop không
- Reproduce pipeline sau này

### 13.3 Xử lý `valid.json` và `test.json`

```
valid.json:
    - Áp dụng Bước 1, 2, 3 (sửa artifact + strip [asy] + decimal norm + extract answer)
    - KHÔNG drop duplicate (cần đủ mẫu để eval)
    - KHÔNG drop conflict (eval vẫn cần biết ground truth)
    - Dùng type để stratified evaluation

test.json:
    - Áp dụng Bước 1, 2 (sửa artifact + strip [asy] + decimal norm + clean nhẹ)
    - KHÔNG rebuild response (không có ground truth)
    - KHÔNG drop bất kỳ mẫu nào
    - [asy] trong test: strip như bình thường — phần text mô tả bài toán vẫn còn đủ
      để model giải; không lo mất thông tin vì [asy] không mang signal giải toán
```

### 13.4 Những rủi ro cần theo dõi

| Rủi ro | Dấu hiệu | Cách xử lý |
|---|---|---|
| Model không sinh ra "Đáp án là:" | Score rất thấp toàn bộ | Kiểm tra lại anchor consistency |
| Model truncate giữa chừng | Output không có dòng cuối | Kiểm tra MAX_LENGTH, giảm truncation |
| Nhóm MATH score thấp bất thường | MATH F1 << GSM F1 | Kiểm tra LaTeX preservation, [asy] handling |
| Score FOBAR/SV thấp | Dạng bài tìm biến x | Model chưa đủ data — đây là dạng khó |
| Score không cải thiện sau epoch 2 | Loss plateau | Kiểm tra labels mask, xem có tính loss trên prompt không |

### 13.5 Reproducibility

```python
# Đặt seed trước mọi thao tác random
import random
random.seed(42)

# Lưu hash của dataset sau preprocessing để verify sau này
import hashlib, json

def dataset_fingerprint(dataset):
    content = json.dumps(
        [r["query_clean"] + r["response_clean"] for r in dataset],
        ensure_ascii=False
    ).encode()
    return hashlib.md5(content).hexdigest()
```

---

## 14. Checklist trước khi train

Chạy toàn bộ các check sau và đảm bảo kết quả pass trước khi bắt đầu training:

```
[ ] Tổng mẫu sau preprocessing: 97,500 – 98,700
[ ] 100% mẫu có anchor "Đáp án là:" là dòng cuối
[ ] 0% mẫu còn anchor "Câu trả lời là:"
[ ] 0% exact duplicate
[ ] LaTeX preservation ratio > 0.99
[ ] 0% còn \đóng hộp{
[ ] 0% mẫu có token length > MAX_LENGTH
[ ] Không nhóm type nào mất > 5% mẫu
[ ] labels[prompt_positions] tất cả = -100
[ ] labels[pad_positions] tất cả = -100
[ ] pad_token_id = eos_token_id = 50256
[ ] Log file cho extract_failed, exact_duplicates, conflict_dropped, too_long_dropped đã được lưu
[ ] valid.json đã được xử lý riêng (không drop, không rebuild)
[ ] Dataset fingerprint đã được lưu để verify reproducibility
```

---

## Phụ lục A — Quyết định thiết kế và lý do

| Quyết định | Lý do |
|---|---|
| Dùng `query_vi`, không dùng `original_question_vi` | 60,619 mẫu đã augmentation — original không phải là input đúng |
| Anchor duy nhất `Đáp án là:` | Scoring phụ thuộc extract cuối, model cần học 1 pattern nhất quán |
| Majority vote thay vì drop toàn bộ conflict | Giữ được ~106 mẫu hợp lệ, ít mất data hơn |
| Smart truncation: cắt giữa, giữ cuối | Dòng đáp án cuối là tín hiệu quan trọng nhất, không bao giờ cắt |
| Không xóa `[asy]` mặc định | Test set có thể chứa dạng này; chỉ drop nếu quá dài |
| Template ngắn `Câu hỏi: / Lời giải:` | Ít overhead token hơn `### Bài toán: / ### Lời giải:` cho GPT-2 nhỏ |
| Dynamic padding thay vì static | Hiệu quả memory và throughput tốt hơn với Hugging Face DataCollator |
| `labels[prompt] = -100` | Không lãng phí capacity model vào việc tái sinh câu hỏi |
| Decimal comma → dấu chấm (chỉ dòng cuối) | Script extract dùng regex số, dấu phẩy gây false negative |
| Giữ LaTeX trong lời giải | 33,685 mẫu MATH cần ký hiệu toán; ép về text phá target |

---

*Tài liệu này tổng hợp phân tích từ `data_understanding.md` và `preprocessing_proposal.md`, kết hợp điểm mạnh của cả hai hướng preprocessing và bổ sung các quyết định thiết kế tối ưu cho bài toán fine-tune GPT-2 giải toán tiếng Việt.*