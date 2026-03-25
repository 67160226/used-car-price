import streamlit as st
import pandas as pd
import joblib
import json

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Cardekho Price Predictor", page_icon="🏎️", layout="centered")

# 1. โหลดโมเดลและข้อมูลประกอบ (Metadata)
@st.cache_resource
def load_model_data():
    try:
        # แก้ไข path ให้โหลดไฟล์จากโฟลเดอร์เดียวกันโดยตรง
        model = joblib.load('car_price_pipeline.pkl')
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return model, metadata
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการโหลดไฟล์โมเดล: {e}")
        st.info("💡 คำแนะนำ: หากเจอ Error '_RemainderColsList' แปลว่าเวอร์ชัน scikit-learn ไม่ตรงกัน แนะนำให้อัปโหลดขึ้น Streamlit Cloud ตามวิธีที่แนะนำไปครับ")
        st.stop()

model, meta = load_model_data()

# 2. ส่วนหัวของเว็บ
st.title("🏎️ AI ประเมินราคารถยนต์มือสอง")
st.markdown(f"**ขับเคลื่อนโดย:** `{meta['model_used']}` | **ความแม่นยำ (R²):** `{meta['r2_score']:.2%}`")
st.write("กรอกสเปกรถยนต์ของคุณด้านล่าง เพื่อให้ AI วิเคราะห์ราคาขายที่เหมาะสม (อ้างอิงจากฐานข้อมูล Cardekho)")
st.divider()

# 3. ฟอร์มรับข้อมูลจากผู้ใช้งาน
with st.form("car_price_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 ข้อมูลพื้นฐาน")
        brand = st.selectbox("ยี่ห้อรถ (Brand)", meta['brands'])
        vehicle_age = st.number_input("อายุการใช้งาน (ปี)", min_value=0, max_value=25, value=5)
        km_driven = st.number_input("ระยะทางที่วิ่งมาแล้ว (กม.)", min_value=1000, max_value=500000, value=50000, step=5000)
        seller_type = st.selectbox("ประเภทผู้ขาย", meta['seller_types'])

    with col2:
        st.subheader("⚙️ สเปกเครื่องยนต์")
        fuel_type = st.selectbox("ประเภทเชื้อเพลิง", meta['fuel_types'])
        transmission_type = st.selectbox("ระบบเกียร์", meta['transmission_types'])
        engine = st.number_input("ขนาดเครื่องยนต์ (CC)", min_value=600, max_value=6500, value=1200, step=100)
        max_power = st.number_input("แรงม้า (Max Power - bhp)", min_value=30.0, max_value=600.0, value=85.0)
        mileage = st.number_input("อัตราสิ้นเปลือง (kmpl)", min_value=5.0, max_value=40.0, value=18.0)
        seats = st.slider("จำนวนที่นั่ง", min_value=2, max_value=14, value=5)

    # ปุ่มกดประเมิน
    submitted = st.form_submit_button("🔮 ให้ AI ประเมินราคา", use_container_width=True)

# 4. ส่วนแสดงผลลัพธ์การพยากรณ์
if submitted:
    # นำข้อมูลที่กรอก มาจัดรูปแบบเป็น DataFrame ให้ตรงกับตอนเทรนโมเดล
    input_data = pd.DataFrame([{
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
    
    with st.spinner('กำลังคำนวณราคากลาง...'):
        prediction = model.predict(input_data)[0]
    
    st.success("✅ ประเมินราคาเสร็จสิ้น!")
    st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>₹ {prediction:,.0f} รูปีอินเดีย</h2>", unsafe_allow_html=True)
    st.caption("หมายเหตุ: ราคานี้เป็นเพียงการประเมินจากข้อมูลทางสถิติเท่านั้น")
    st.balloons()
