import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from underthesea import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# Danh sách hư từ (function words) tiếng Việt bộc lộ thói quen viết vô thức
VIETNAMESE_FUNCTION_WORDS = [
    "thì", "là", "mà", "của", "và", "nhưng", "tuy_nhiên", "do_đó",
    "vì_vậy", "bởi_vì", "để", "cho", "với", "trong", "ngoài", "trên",
    "dưới", "các", "những", "một", "này", "đó", "nên", "nếu", "khi",
    "đã", "đang", "sẽ", "cũng", "vẫn", "đều", "rất", "quá", "lắm"
]

PUNCTUATIONS = [',', ';', ':', '-', '(', ')', '"', '?', '!']


class AuthorshipAnalyzer:
    def __init__(self, model_name: str = "vinai/phobert-base-v2"):
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True)
        base_model.eval()
        # Nén trọng số INT8 trên CPU để chạy mượt trong giới hạn 512MB RAM của Render
        self.model = torch.quantization.quantize_dynamic(
            base_model, {torch.nn.Linear}, dtype=torch.qint8
        )

    def extract_stylometry(self, text: str) -> dict:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        segmented_text = word_tokenize(text, format="text").lower()
        tokens = segmented_text.split()

        total_words = max(len(tokens), 1)
        total_chars = max(len(text), 1)
        total_sents = max(len(sentences), 1)

        # 1. Chỉ số thống kê bề mặt
        avg_word_len = sum(len(w.replace("_", "")) for w in tokens) / total_words
        avg_sent_len = total_words / total_sents
        ttr = len(set(tokens)) / total_words  # Type-Token Ratio (Độ phong phú từ vựng)

        # 2. Tần suất dấu câu
        punct_freqs = [text.count(p) / total_chars for p in PUNCTUATIONS]

        # 3. Tần suất hư từ
        func_freqs = [tokens.count(fw) / total_words for fw in VIETNAMESE_FUNCTION_WORDS]

        # Vector chuẩn hóa (scale nhẹ các chỉ số lớn để cân bằng khi tính cosine)
        feature_vector = np.array(
            [avg_word_len / 10.0, avg_sent_len / 50.0, ttr] + punct_freqs + func_freqs,
            dtype=np.float32
        )

        # Tìm top 5 hư từ tác giả hay dùng nhất để hiển thị lên giao diện
        func_counts = {fw.replace("_", " "): tokens.count(fw) for fw in VIETNAMESE_FUNCTION_WORDS if tokens.count(fw) > 0}
        top_func_words = sorted(func_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "vector": feature_vector,
            "metrics": {
                "total_words": total_words,
                "total_sentences": total_sents,
                "avg_word_len": round(avg_word_len, 2),
                "avg_sent_len": round(avg_sent_len, 2),
                "ttr": round(ttr * 100, 1),
                "top_func_words": top_func_words
            }
        }

    def extract_phobert_embedding(self, text: str, max_len: int = 256, stride: int = 128) -> np.ndarray:
        segmented_text = word_tokenize(text, format="text")
        encoded = self.tokenizer(
            segmented_text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False
        )
        input_ids = encoded["input_ids"][0]

        if len(input_ids) == 0:
            return np.zeros(self.model.config.hidden_size, dtype=np.float32)

        # Cắt đoạn Sliding Window để xử lý bài báo dài hơn 256 tokens
        chunks = []
        chunk_size = max_len - 2  # Dành 2 vị trí cho <s> và </s>
        for i in range(0, len(input_ids), stride):
            chunk = input_ids[i:i + chunk_size]
            chunk = torch.cat([
                torch.tensor([self.tokenizer.cls_token_id]),
                chunk,
                torch.tensor([self.tokenizer.sep_token_id])
            ])
            chunks.append(chunk)
            if i + chunk_size >= len(input_ids):
                break

        chunk_embeddings = []
        with torch.no_grad():
            for chunk in chunks:
                ids = chunk.unsqueeze(0).to(self.device)
                mask = torch.ones_like(ids).to(self.device)
                outputs = self.model(input_ids=ids, attention_mask=mask)
                # Mean Pooling
                emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
                chunk_embeddings.append(emb)

        return np.mean(chunk_embeddings, axis=0)

    def compare_articles(self, ref_text: str, target_text: str) -> dict:
        ref_style = self.extract_stylometry(ref_text)
        target_style = self.extract_stylometry(target_text)

        ref_bert = self.extract_phobert_embedding(ref_text)
        target_bert = self.extract_phobert_embedding(target_text)

        # Tính độ tương đồng Cosine cho từng nhánh
        style_sim = float(cosine_similarity([ref_style["vector"]], [target_style["vector"]])[0][0])
        bert_sim = float(cosine_similarity([ref_bert], [target_bert])[0][0])

        # Chuẩn hóa về thang [0, 1]
        style_score = max(0.0, min(1.0, style_sim))
        # PhoBERT cosine thường nằm trong khoảng [0.65, 1.0] với văn bản tiếng Việt, kéo giãn dải điểm để phân tách rõ hơn
        bert_score = max(0.0, min(1.0, (bert_sim - 0.60) / 0.40))

        # Trọng số lai: 55% Stylometry (phong cách viết cốt lõi) + 45% PhoBERT (ngữ cảnh & cấu trúc sâu)
        final_score = round((0.55 * style_score + 0.45 * bert_score) * 100, 1)

        if final_score >= 72.0:
            verdict = "KHỚP TÁC GIẢ"
            status = "match"
            explanation = "Phong cách sử dụng hư từ, nhịp điệu câu và cấu trúc ngữ nghĩa giữa hai bài báo có độ tương đồng cao."
        elif final_score >= 52.0:
            verdict = "NGHI VẤN / CẦN KIỂM TRA THÊM"
            status = "warning"
            explanation = "Có sự sai lệch nhất định về nhịp câu hoặc cách dùng từ chức năng. Có thể bài viết đã bị biên tập lại nhiều hoặc đồng tác giả."
        else:
            verdict = "SAI TÁC GIẢ (CẢNH BÁO CAO)"
            status = "mismatch"
            explanation = "Dấu vân tay ngôn ngữ khác biệt rõ rệt về tần suất hư từ, độ dài câu và cấu trúc diễn đạt so với tác giả gốc."

        return {
            "final_score": final_score,
            "style_score": round(style_score * 100, 1),
            "bert_score": round(bert_score * 100, 1),
            "verdict": verdict,
            "status": status,
            "explanation": explanation,
            "ref_metrics": ref_style["metrics"],
            "target_metrics": target_style["metrics"]
        }