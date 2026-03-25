import joblib
import json
import os
import sklearn # นำเข้า sklearn เพื่อเช็คเวอร์ชัน

# 1. ตรวจสอบก่อนว่ามีตารางสรุปผล (df_results) หรือไม่
if 'df_results' in locals():
    print("--- 🎯 กำลังคัดเลือกโมเดลที่ดีที่สุด (Cardekho Edition) ---")

    # ค้นหาโมเดลที่ได้ R2 Score สูงที่สุด
    best_model_info = df_results.iloc[0]
    best_model_name = best_model_info['Model']
    best_r2 = best_model_info['R2 Score']

    print(f"✅ โมเดลที่ชนะเลิศคือ: {best_model_name}")
    print(f"📊 ค่าความแม่นยำ (R2 Score): {best_r2:.4f}")

    # 2. คัดเลือก Pipeline ของโมเดลที่ชนะ
    best_model_obj = next(m for name, m in models if name == best_model_name)

    # สร้าง Pipeline สุดท้ายเพื่อบันทึก
    best_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', best_model_obj)
    ])

    # เทรนโมเดลตัวที่เก่งที่สุดอีกครั้งด้วยข้อมูลทั้งหมด
    best_pipeline.fit(X_train, y_train)

    # 3. บันทึกไฟล์ลงในโฟลเดอร์ model_artifacts
    os.makedirs('model_artifacts', exist_ok=True)
    
    # บันทึกโมเดล (.pkl)
    model_path = 'model_artifacts/car_price_pipeline.pkl'
    joblib.dump(best_pipeline, model_path)

    # 4. บันทึกข้อมูลประกอบ (.json) 
    metadata = {
        "model_used": best_model_name,
        "r2_score": float(best_r2),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "brands": sorted(df_clean['brand'].unique().tolist()),
        "seller_types": sorted(df_clean['seller_type'].unique().tolist()),
        "fuel_types": sorted(df_clean['fuel_type'].unique().tolist()),
        "transmission_types": sorted(df_clean['transmission_type'].unique().tolist()),
        "sklearn_version": sklearn.__version__ # 👈 เพิ่มการบันทึกเวอร์ชันตรงนี้!
    }
    
    with open('model_artifacts/model_metadata.json', 'w') as f:
        json.dump(metadata, f)

    print(f"📂 บันทึกไฟล์สำเร็จ!")
    print(f"1. {model_path}")
    print(f"2. model_artifacts/model_metadata.json")
    print("\n⚠️ สำคัญมากเพื่อป้องกัน Error _RemainderColsList:")
    print(f"ให้แก้ไฟล์ requirements.txt ในคอมพิวเตอร์ของคุณ โดยระบุเวอร์ชันตามนี้ครับ:")
    print(f"scikit-learn=={sklearn.__version__}")

else:
    print("❌ Error: ไม่พบตาราง df_results กรุณารันเซลล์เปรียบเทียบโมเดลก่อนหน้านี้อีกครั้ง")
