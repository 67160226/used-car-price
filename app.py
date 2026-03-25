import streamlit as st
import pandas as pd
import joblib
import json

# ตั้งค่าหน้าเว็บให้ดูทันสมัย
st.set_page_config(page_title="Cardekho Car Price Predictor", page_icon="🏎️", layout="centered")

# 1. ฟังก์ชันโหลดโมเดลและข้อมูลประกอบ
@st.cache_resource
def load_artifacts():
    # โหลดตัวโมเดล Pipeline
    model = joblib.load('model_artifacts/car_price_pipeline.pkl')
    # โหลดค่า Metadata (ยี่ห้อ, เชื้อเพลิง, ความแม่นยำ)
    with open('model_artifacts/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    return model, metadata

try:
    model, meta = load_artifacts()
except Exception as e:
    st.error(f"❌ ไม่พบไฟล์โมเดลในโฟลเดอร์ model_artifacts: {e}")
    st.stop()

# 2. ส่วนหัวของเว็บไซต์
st.title("🏎️ ระบบประเมินราคารถยนต์ Cardekho")
st.markdown(f"**AI Model:** `{meta['model_used']}` | **Accuracy (R²):** `{meta['r2_score']:.2%}`")
st.info("กรอกข้อมูลสเปกรถยนต์ด้านล่างเพื่อคำนวณราคาขาย (หน่วย: รูปีอินเดีย ₹)")

# 3. ส่วนรับข้อมูลจากผู้ใช้ (Inputs)
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 ข้อมูลพื้นฐาน")
        # ดึงรายชื่อยี่ห้อจาก Metadata โดยตรง
        brand = st.selectbox("ยี่ห้อ (Brand)", meta['brands'])
        vehicle_age = st.number_input("อายุรถ (ปี)", min_value=0, max_value=25, value=5)
        km_driven = st.number_input("ระยะทางที่วิ่งมาแล้ว (KM)", min_value=1000, max_value=500000, value=50000, step=5000)
        seller_type = st.selectbox("ประเภทผู้ขาย", meta['seller_types'])

    with col2:
        st.subheader("⚙️ สเปกเครื่องยนต์")
        fuel_type = st.selectbox("ประเภทเชื้อเพลิง", meta['fuel_types'])
        transmission_type = st.selectbox("ระบบเกียร์", meta['transmission_types'])
        engine = st.number_input("ขนาดเครื่องยนต์ (CC)", min_value=600, max_value=6500, value=1200, step=100)
        max_power = st.number_input("แรงม้า (bhp)", min_value=30.0, max_value=600.0, value=85.0)
        mileage = st.number_input("อัตราสิ้นเปลือง (kmpl)", min_value=5.0, max_value=35.0, value=18.0)
        seats = st.slider("จำนวนที่นั่ง", 2, 10, 5)

    submit = st.form_submit_button("🔮 ประเมินราคาขายทันที", use_container_width=True)

# 4. ส่วนแสดงผลลัพธ์
if submit:
    # สร้าง DataFrame ให้ตรงกับที่ AI เคยเรียนรู้มา
    input_df = pd.DataFrame([{
        'vehicle_age': vehicle_age,
        'km_driven': km_driven,
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'seats': seats,
        'brand': brand,
        'seller_type': seller_type,
        'fuel_type': fuel_type,
        'transmission_type': transmission_type
    }])
    
    # พยากรณ์ราคา
    prediction = model.predict(input_df)[0]
    
    st.divider()
    st.balloons()
    st.subheader("💰 ราคาประเมินโดย AI:")
    st.header(f"₹ {prediction:,.0f} รูปี")
    st.caption("หมายเหตุ: ราคานี้เป็นการวิเคราะห์เชิงสถิติจากฐานข้อมูล Cardekho ประเทศอินเดีย")
