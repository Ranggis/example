import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time

# Load model dan encoder
with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# (Opsional) Load akurasi model jika tersedia
try:
    with open("model_accuracy.pkl", "rb") as f:
        accuracy = pickle.load(f)
except:
    accuracy = None

# Sidebar
st.sidebar.success("Selamat datang di Aplikasi Prediksi Prestasi Siswa! 🎓")

st.title("\U0001F393 Prediksi Siswa Berprestasi")
st.markdown("Masukkan informasi siswa di bawah ini untuk memprediksi apakah ia berprestasi atau tidak.")

# Form input
with st.form("input_form"):
    name = st.text_input("Nama Siswa")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("👤 Jenis Kelamin", label_encoders['gender'].classes_)
        race = st.selectbox("🌎 Ras/Etnis", label_encoders['race/ethnicity'].classes_)
        parental_edu = st.selectbox("🎓 Pendidikan Orang Tua", label_encoders['parental level of education'].classes_)

    with col2:
        lunch = st.selectbox("🍱 Tipe Makan Siang", label_encoders['lunch'].classes_)
        test_prep = st.selectbox("📚 Kursus Persiapan Tes", label_encoders['test preparation course'].classes_)
        math_score = st.number_input("🧮 Skor Matematika", min_value=0, max_value=100, value=70)
        reading_score = st.number_input("📖 Skor Membaca", min_value=0, max_value=100, value=70)
        writing_score = st.number_input("✍️ Skor Menulis", min_value=0, max_value=100, value=70)

    submitted = st.form_submit_button("🔍 Prediksi")

# Proses prediksi jika disubmit
if submitted:
    with st.spinner("⏳ Memproses prediksi..."):
        time.sleep(1)

        input_dict = {
            'gender': label_encoders['gender'].transform([gender])[0],
            'race/ethnicity': label_encoders['race/ethnicity'].transform([race])[0],
            'parental level of education': label_encoders['parental level of education'].transform([parental_edu])[0],
            'lunch': label_encoders['lunch'].transform([lunch])[0],
            'test preparation course': label_encoders['test preparation course'].transform([test_prep])[0],
            'math score': math_score,
            'reading score': reading_score,
            'writing score': writing_score
        }

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]

        # Output hasil prediksi
        if prediction == 1:
            st.success(f"\U0001F31F {name} diprediksi **Berprestasi**!")
        else:
            st.warning(f"❌ {name} diprediksi **Tidak Berprestasi**.")

        # Penjelasan
        st.markdown("### 📘 Keterangan:")
        st.markdown("""
        - **Berprestasi**: Siswa memiliki rata-rata skor yang tinggi dan/atau memenuhi pola yang sesuai dengan siswa-siswa berprestasi sebelumnya.
        - **Tidak Berprestasi**: Siswa mungkin memerlukan dukungan tambahan untuk meningkatkan performa akademik.
        """)

        # Rata-rata skor dan insight
        avg_score = (math_score + reading_score + writing_score) / 3
        st.info(f"📊 Rata-rata skor {name}: {avg_score:.2f}")

        # Donut Chart visualisasi skor
        st.subheader("📊 Proporsi Skor Akademik")
        scores = {'Matematika': math_score, 'Membaca': reading_score, 'Menulis': writing_score}
        fig, ax = plt.subplots()
        labels = list(scores.keys())
        sizes = list(scores.values())
        colors = ['#ff9999','#66b3ff','#99ff99']
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,
                                          startangle=90, wedgeprops={'width':0.4})
        ax.axis('equal')
        st.pyplot(fig)

        # Tombol prediksi ulang
        if st.button("🔄 Prediksi Siswa Lain"):
            st.experimental_rerun()

# Footer
st.markdown("""
---
📌 *Model ini menggunakan algoritma K-Nearest Neighbors untuk memprediksi potensi prestasi siswa berdasarkan data input. 
Gunakan hasil prediksi sebagai alat bantu, bukan keputusan akhir.*

👨‍💻 Dibuat oleh: Ranggis
""")

# Tampilkan akurasi jika ada
if accuracy is not None:
    st.caption(f"🎯 Akurasi Model: {accuracy*100:.2f}%")
