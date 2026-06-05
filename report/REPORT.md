# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyen Van Minh
**Nhóm:** A2
**Ngày:** 5/6/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**


Cosine similarity là đo độ tương đồng của 2 vector trong không gian embedding nhiều chiều
Quá trình huấn luyện mô hình NLP làm cho các vector của các từ cùng ý nghĩa gần nhau hơn về góc
cụ thể là về tính toán P(word_a | word_b) dựa vào công thức Softmax trong đó sử dụng dot product
dot(a,b) = |a||b|cos(theta) vậy nên khi 2 từ/câu càng gần nghĩa thì góc của chúng càng nhỏ -> Cosine càng gần về 1

VD:
- Chú mèo dễ thương
- Những chú chó đốm
2 câu trên sẽ khá gần nhau trong không gian vector vì cùng chủ đề động vật

VD:
- Chú mèo dễ thương
- Ông Kim Jongun vừa thả bom
2 câu trên sẽ có cosine thấp ~0

> Euclide ít dùng hơn vì chúng bị ảnh hưởng bởi khoảng cách 2 điểm, dù góc rất nhỏ nhưng độ dài vector khác nhau dẫn 
đế kết quả so sánh khác nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
chunk_num = char_num / (chunk_size - overlap) 
> *Đáp án:*
~ 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:*
overlap tang thi chunk count tang
> Overlap tang de thong tin cua moi chunk duoc smooth hon ve mat ngu nghia

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Corporate News

**Tại sao nhóm chọn domain này?**
> Nhóm lựa chọn domain Corporation Policy vì các tài liệu liên quan đến hoạt động doanh nghiệp, thay đổi nhân sự, đầu tư và quản trị thường chứa nhiều thông tin có cấu trúc rõ ràng như tên tổ chức, chức vụ, thời gian và sự kiện. Đây là loại dữ liệu phù hợp để đánh giá hiệu quả của các chiến lược chunking và retrieval do yêu cầu truy xuất chính xác các thông tin cụ thể từ văn bản dài. Ngoài ra, metadata như thời gian, danh mục và nguồn tin cũng hỗ trợ tốt cho việc lọc và tìm kiếm.

### Data Inventory

| # | Tên tài liệu                                          | Nguồn      | Số ký tự     | Metadata đã gán                  |
| - | ----------------------------------------------------- | ---------- | ------------ | -------------------------------- |
| 1 | Novaland có tân Tổng giám đốc                         | VnEconomy  | ~3,000       | url, title, time, category, tags |
| 2 | Miền Nam dư thừa nhiều loại nông sản                  | VnEconomy  | ~2,500       | url, title, time, category, tags |
| 3 | Điểm mặt những mảnh đất vàng độc nhất vô nhị ở Hà Nội | Báo Đầu Tư | ~4,000       | url, title, time, category, tags |
| 4 | Các bài báo doanh nghiệp khác trong tập dữ liệu       | VnEconomy  | ~2,000–5,000 | url, title, time, category, tags |
| 5 | Các bài báo doanh nghiệp khác trong tập dữ liệu       | Báo Đầu Tư | ~2,000–5,000 | url, title, time, category, tags |

### Metadata Schema

| Trường metadata | Kiểu        | Ví dụ giá trị                                     | Tại sao hữu ích cho retrieval?           |
| --------------- | ----------- | ------------------------------------------------- | ---------------------------------------- |
| title           | string      | Novaland có tân Tổng giám đốc                     | Cung cấp ngữ cảnh tổng quát của tài liệu |
| category        | string      | Chứng khoán                                       | Hỗ trợ lọc theo lĩnh vực nội dung        |
| time            | date/string | 2024-11-04                                        | Giúp truy xuất thông tin theo thời gian  |
| url             | string      | [https://vneconomy.vn/](https://vneconomy.vn/)... | Hỗ trợ truy vết nguồn gốc tài liệu       |
| tags            | string/list | novaland, nhân sự, CEO                            | Tăng khả năng matching theo từ khóa      |
| source_doc      | string      | news_2                                            | Xác định tài liệu gốc của chunk          |
| chunk_index     | integer     | 3                                                 | Hỗ trợ truy vết chunk được retrieve      |


---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Document ID: 0 - Title: Novaland có tân Tổng giám đốc| FixedSizeChunker (`fixed_size`) |14 | 197| yes|
| | SentenceChunker (`by_sentences`) | 5| 420| yes|
| | RecursiveChunker (`recursive`) | 58| 35| no|
| Document ID: 1 - Title: Miền Nam dư thừa nhiều loại nông sản| FixedSizeChunker (`fixed_size`) |35 | 199| yes|
| | SentenceChunker (`by_sentences`) | 11| 4478| yes|
| | RecursiveChunker (`recursive`) | 394| 12| no|


### Strategy Của Tôi

**Loại:** Proposition-based Chunking

Mô tả cách hoạt động

Tài liệu được chuyển thành các atomic propositions bằng LLM. Sau đó các proposition được gom nhóm thành các chunk dựa trên mức độ liên quan về ngữ nghĩa. Mỗi khi một proposition mới được thêm vào chunk, hệ thống cập nhật lại summary và title của chunk để phản ánh nội dung hiện tại.

Tại sao tôi chọn strategy này cho domain nhóm?

Dữ liệu tin tức thường chứa nhiều sự kiện, số liệu và thực thể khác nhau trong cùng một bài viết. Proposition Chunking giúp tách các thông tin này thành các đơn vị tri thức nhỏ trước khi tái tổ chức thành các nhóm thông tin có liên quan, từ đó cải thiện khả năng retrieval cho các câu hỏi chi tiết.

**Code snippet (nếu custom):**
```python
# Paste implementation here
doc -> split to paragraphs

def add_propositions(paragraphs):
    Loop in existing chunk:
        LLM find best match summaried chunk
        1. matched -> append
        2. not found -> create new chunk
        LLM summary new chunk

def _update_chunk_title(chunk)
    LLM + prompt

def _update_chunk_new_summary(chunk):
    LLM + prompt

def _find_relevant_chunk(chunk):
    LLM + prompt
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | Fixed_size| 20-30| Kha tot|
| | **của tôi** | Proposition-based| 2-3| Te, chunk size qua lon|

### So Sánh Với Thành Viên Khác

| Strategy                | Retrieval  | Chunk Coherence | Chunk Size Control |
| ----------------------- | ---------- | --------------- | ------------------ |
| Semantic                | Trung bình | Cao             | Tốt                |
| Agentic                 | Cao        | Cao             | Trung bình         |
| Proposition-based (tôi) | Cao        | Rất cao         | Thấp               |
| Document Structure      | Cao nhất   | Cao             | Rất tốt            |


**Strategy nào tốt nhất cho domain này? Tại sao?**

> Document Structure Chunking đạt kết quả retrieval tốt nhất trên bộ dữ liệu tin tức của nhóm. Tuy nhiên Proposition-based Chunking cho thấy khả năng gom nhóm thông tin theo ý nghĩa rất tốt và tạo ra các chunk có semantic coherence cao nhất. Hạn chế chính là khó kiểm soát kích thước chunk và chi phí xử lý lớn do phụ thuộc nhiều vào LLM.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Split bang regex lib, chua co filter ki tu la. Split = ".,?!..."

**`RecursiveChunker.chunk` / `_split`** — approach:
> Split theo ["\n\n", "\n", ".", " "]

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Document (content, source, id) + embedding local -> chromedb 

**`search_with_filter` + `delete_document`** — approach:
> Chua co filter category

### KnowledgeBaseAgent

**`answer`** — approach:
> Prompt: basic, answer based on document context (no guardrails...) 

### Test Results

```
# Paste output of: pytest tests/ -v
```

**Số tests pass:** 42 passed in 0.71s / 42
```
plugins: langsmith-0.4.32, anyio-4.10.0
collected 42 items                                                                                                                                                             

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                                                    [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                                                             [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                                                      [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                                                       [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                                                            [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                                                            [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                                                                  [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                                                                   [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                                                                 [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                                                                   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                                                                   [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                                                              [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                                                          [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                                                    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                                                           [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                                                               [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                                                         [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                                                               [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                                                                   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                                                     [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                                                       [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                                                             [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                                                                  [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                                                    [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                                                        [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                                                     [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                                                              [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                                                             [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                                                        [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                                                    [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                                                               [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                                                                   [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                                                         [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                                                                   [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                                                                [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                                                              [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                                                             [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                                                                 [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                                                            [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                                                     [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                                                           [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                                                               [100%]

============================================================================== 42 passed in 0.71s ==============================================================================
```
---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | | | high / low | | |
| 2 | | | high / low | | |
| 3 | | | high / low | | |
| 4 | | | high / low | | |
| 5 | | | high / low | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

> Benchmark tren QA (10 cau)

FILE: [data/benchmark.md]

### Kết Quả Của Tôi

**Findings**:

"Trong Proposition-based Chunking, việc liên tục cập nhật summary của chunk vô tình tạo ra hiệu ứng snowball, khiến LLM ngày càng ưu tiên đưa proposition mới vào các chunk đã tồn tại thay vì tạo chunk mới. Điều này làm semantic coherence tăng nhưng chunk size trở nên khó kiểm soát."


---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 4 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 10 / 15 |
| My approach | Cá nhân | 7 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 7 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | ** 76 / 100** |
