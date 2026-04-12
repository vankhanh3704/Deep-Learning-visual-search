import streamlit as st
import requests
from PIL import Image
import os
import random

# ==========================================
# CAU HINH HE THONG
# ==========================================
API_URL = "http://localhost:8000/search/"

st.set_page_config(
    page_title="Shopee AI Visual Search", 
    page_icon="🛍️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS NHUNG VAO STREAMLIT
# ==========================================
st.markdown("""
<style>
    /* Chinh mau nen va font chu */
    .stApp {
        background-color: #f5f5f5;
    }
    
    /* Style cho Tieu de chinh */
    .main-header {
        text-align: center;
        color: #ee4d2d;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: -10px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* Style cho Nut bam Shopee */
    .stButton>button {
        background-color: #ee4d2d !important;
        color: white !important;
        border-radius: 4px;
        border: none;
        width: 100%;
        padding: 10px 0;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #d73d1f !important;
        box-shadow: 0 4px 8px rgba(238, 77, 45, 0.3);
        transform: translateY(-2px);
    }

    /* Style cho the San pham (Product Card) */
    .product-card {
        background-color: white;
        border-radius: 4px;
        padding: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        margin-bottom: 15px;
        height: 100%;
    }
    
    .product-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        border: 1px solid #ee4d2d;
    }
    
    .product-title {
        font-size: 13px;
        color: #333;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
        margin: 10px 0 5px 0;
        line-height: 1.4;
    }
    
    .product-price {
        color: #ee4d2d;
        font-weight: bold;
        font-size: 16px;
        margin: 0;
    }
    
    .product-stats {
        font-size: 11px;
        color: #757575;
        display: flex;
        justify-content: space-between;
        margin-top: 5px;
    }
    
    .similarity-badge {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 2px 6px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: bold;
        display: inline-block;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# GIAO DIEN CHINH
# ==========================================
st.markdown("<h1 class='main-header'>🛍️ Shopee Visual Search</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Hệ thống truy xuất sản phẩm tương tự bằng Trí tuệ Nhân tạo</p>", unsafe_allow_html=True)

# Layout chia làm 3 cột để căn giữa ô upload
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader("📸 Tải lên ảnh sản phẩm bạn muốn tìm...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Hien thi anh da tai len can giua
        st.markdown("<div style='display: flex; justify-content: center; margin-bottom: 15px;'>", unsafe_allow_html=True)
        st.image(uploaded_file, caption="Ảnh truy vấn", width=300)
        st.markdown("</div>", unsafe_allow_html=True)
        
        search_clicked = st.button("🔍 Tìm kiếm Sản phẩm Tương tự")

# ==========================================
# XU LY LOGIC TIM KIEM
# ==========================================
if uploaded_file is not None and search_clicked:
    st.markdown("---")
    
    with st.spinner("⏳ AI đang trích xuất đặc trưng và quét kho dữ liệu..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    inf_time = data.get('inference_time', 0)
                    st.success(f"✨ Tìm kiếm thành công trong {inf_time:.3f} giây! Dưới đây là các sản phẩm tương tự:")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Hien thi 5 ket qua
                    cols = st.columns(5)
                    for idx, result in enumerate(data["results"]):
                        img_path = result["image_path"]
                        distance = result["distance"]
                        
                        # Tao data gia lap cho giong Shopee
                        fake_price = f"{random.randint(50, 499)}.000 ₫"
                        fake_sold = f"Đã bán {random.randint(1, 9)}.{random.randint(1, 9)}k"
                        fake_title = "Sản phẩm Thời trang / Gia dụng cao cấp mẫu mới nhất"
                        similarity = max(0, 100 - (distance * 15)) # Tinh phan tram giong nhau
                        
                        with cols[idx]:
                            if os.path.exists(img_path):
                                res_img = Image.open(img_path)
                                
                                # Tao the HTML Card hien thi san pham
                                st.image(res_img, use_container_width=True)
                                st.markdown(f"""
                                    <div class="product-card">
                                        <p class="product-title">{fake_title}</p>
                                        <p class="product-price">{fake_price}</p>
                                        <div class="product-stats">
                                            <span>⭐⭐⭐⭐⭐</span>
                                            <span>{fake_sold}</span>
                                        </div>
                                        <div class="similarity-badge">Độ giống: {similarity:.1f}%</div>
                                        <p style="font-size: 10px; color: #bdbdbd; margin: 3px 0 0 0;">L2 Dist: {distance:.3f}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.error("Lỗi: Không tìm thấy ảnh gốc.")
                else:
                    st.error(f"Lỗi hệ thống: {data['message']}")
            else:
                st.error(f"Lỗi Server API: {response.status_code}")
        except Exception as e:
            st.error(f"Lỗi kết nối Backend. Hãy chắc chắn bạn đã bật FastAPI. Chi tiết: {e}")