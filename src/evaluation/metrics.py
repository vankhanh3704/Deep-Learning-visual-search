import numpy as np

def calculate_recall_at_k(actual_matches, predicted_matches, k=5):
    """
    Tính Recall@K: Tỉ lệ phần trăm các kết quả đúng nằm trong top K trả về.
    actual_matches: list các ID sản phẩm thực tế giống nhau.
    predicted_matches: list các ID sản phẩm mô hình dự đoán.
    """
    predicted_top_k = predicted_matches[:k]
    hits = len(set(actual_matches) & set(predicted_top_k))
    return hits / len(actual_matches) if len(actual_matches) > 0 else 0.0

def calculate_map(actual_matches, predicted_matches):
    """
    Tính mAP (Mean Average Precision): Trung bình độ chính xác tại từng điểm tìm thấy kết quả đúng.
    """
    hits = 0
    sum_precs = 0
    for i, p in enumerate(predicted_matches):
        if p in actual_matches and p not in predicted_matches[:i]:
            hits += 1
            sum_precs += hits / (i + 1.0)
    
    return sum_precs / len(actual_matches) if len(actual_matches) > 0 else 0.0

# Code test nhanh hàm
if __name__ == "__main__":
    actual = ["A", "B", "C"]
    predicted = ["A", "D", "B", "E", "F"]
    print(f"Recall@5: {calculate_recall_at_k(actual, predicted, k=5):.4f}")
    print(f"mAP: {calculate_map(actual, predicted):.4f}")