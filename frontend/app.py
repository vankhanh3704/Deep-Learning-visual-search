import streamlit as st
import requests
from PIL import Image
import os
import base64

API_URL = "http://localhost:8000/search/"

st.set_page_config(page_title="Shopee Visual Search", layout="wide")

st.markdown("<h1 style='text-align: center; color: #ee4d2d;'>Shopee Visual Search</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Tim kiem san pham tuong tu bang tri tue nhan tao</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Tai len anh san pham ban muon tim kiem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Anh truy van (Anchor)", width=250)
    
    if st.button("Tim kiem san pham tuong tu"):
        with st.spinner("Dang trich xuat dac trung va quet kho du lieu..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    if data["status"] == "success":
                        inf_time = data.get('inference_time', 0)
                        st.success(f"Tim kiem thanh cong trong {inf_time:.3f} giay! Day la cac san pham tuong tu:")
                        
                        cols = st.columns(5)
                        for idx, result in enumerate(data["results"]):
                            img_path = result["image_path"]
                            distance = result["distance"]
                            
                            with cols[idx]:
                                if os.path.exists(img_path):
                                    res_img = Image.open(img_path)
                                    st.image(res_img, use_container_width=True)
                                    # Hien thi phong cach Shopee
                                    st.markdown(f"""
                                        <div style="border: 1px solid #e1e1e1; padding: 10px; border-radius: 5px; margin-top: -10px;">
                                            <p style="color: #ee4d2d; font-weight: bold; margin: 0;">₫99.000</p>
                                            <p style="font-size: 12px; color: gray; margin: 0;">Do giong: {100 - (distance*10):.1f}%</p>
                                            <p style="font-size: 11px; color: #00bfa5; margin: 0;">Khoang cach L2: {distance:.3f}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.warning("Khong tim thay file anh goc.")
                    else:
                        st.error(f"Loi he thong: {data['message']}")
                else:
                    st.error(f"Loi Server: {response.status_code}")
            except Exception as e:
                st.error(f"Loi ket noi Backend. Chi tiet: {e}")