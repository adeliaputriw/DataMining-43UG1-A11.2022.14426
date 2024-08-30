import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


st.write("""
# Klasifikasi Pangkat ASN Kabupaten Sukoharjo (Web Apps)
Aplikasi berbasis Web untuk mengklasifikasi pangkat ASN di **Kabupaten Sukoharjo**.
Data didapat dari rekan kenalan secara pribadi
         """)

st.sidebar.header('Parameter Inputan')

# TAMBAHAN IMAGE ASN
img = Image.open('asn.jpeg')
img = img.resize((700, 418)) #agar gambar tidak terlalu besar
st.image(img, use_column_width=False)
st.write("""
Menurut Peraturan Pemerintah Nomor 11 tahun 2017, pangkat PNS adalah kedudukan yang menunjukan tingkatan jabatan berdasarkan tingkat kesulitan, tanggung jawab, dampak, dan persyaratan kualifikasi pekerjaan.
Nantinya, klasifikasi pangkat ini akan digunakan sebagai dasar perhitungan gaji, tunjangan, dan fasilitas. \n
Gambar diambil dari : https://menpan.go.id/site/berita-terkini/hari-pertama-masuk-kerja-menteri-panrb-akan-pantau-kehadiran-asn
         """)


# NILAI DEFAULT INPUTAN
thn_pengangkatan_default = 2005.0
usia_asn_default = 45.0
masa_kerja_default = 18.0

# UPLOAD FILE CSV
upload_file = st.sidebar.file_uploader("Upload file CSV Anda", type=["csv"])
if upload_file is not None:
    inputan = pd.read_csv(upload_file)

else:
    def input_user():
        pend_terakhir = st.sidebar.selectbox('Pendidikan Terakhir',('D3','D4','S1','S2'))
        
        # INPUT MANUAL DENGAN NILAI DEFAULT
        thn_pengangkatan = st.sidebar.text_input('Tahun Pengangkatan (tahun)', thn_pengangkatan_default)
        usia_asn = st.sidebar.text_input('Usia ASN (tahun)', usia_asn_default)
        masa_kerja = st.sidebar.text_input('Masa Kerja (tahun)', masa_kerja_default)
        
        # KONVERSI INPUT KE TIPE DATA NUMERIK
        thn_pengangkatan = float(thn_pengangkatan) if thn_pengangkatan else thn_pengangkatan_default
        usia_asn = float(usia_asn) if usia_asn else usia_asn_default
        masa_kerja = float(masa_kerja) if masa_kerja else masa_kerja_default
        
        data = {'pend_terakhir' : pend_terakhir,
                'thn_pengangkatan' : thn_pengangkatan,
                'usia_asn' : usia_asn,
                'masa_kerja' : masa_kerja }
        fitur = pd.DataFrame(data, index=[0]) # UNTUK MENYIMPAN DATA
        return fitur
    inputan = input_user()

# MENGGABUNGKAN INPUTAN DAN DATASET PANGKAT   
pangkat_raw = pd.read_csv('pangkat.csv')
pangkatt = pangkat_raw.drop(columns=['pangkat'])
df = pd.concat([inputan, pangkatt], axis=0)

# ENCODE UNTUK FITUR ORDINAL
encode = ['pend_terakhir']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1] #ambil baris pertama (input data user)

# MENAMPILKAN PARAMETER HASIL INPUTAN
st.subheader('Parameter Inputan')


if upload_file is not None:
    st.write(df)
else:
    st.write('Menunggu file csv untuk diupload. Saat ini memakai sampel inputan (seperti tampilan di bawah):')
    st.write(df)
    
# LOAD MODEL NAIVE BAYES CLASSIFIER (NBC)
load_model = pickle.load(open('modelNBC_pangkat.pkl','rb'))

# TERAPKAN NBC NAIVE BAYES CLASSIFIER (NBC)
prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

st.subheader('Keterangan Label Kelas')
jenis_pangkat = np.array(['I/d', 'II/a', 'II/b', 'II/c', 'II/d', 'III/a', 'III/b', 'III/c', 'III/d', 'IV/a', 'IV/b', 'IV/c', 'IV/d'])
label_df = pd.DataFrame({'Kode Pangkat': np.arange(len(jenis_pangkat)), 'Jenis Pangkat': jenis_pangkat})

st.table(label_df)


# MENAMPILKAN HASIL PROBABILITAS HASIL PREDIKSI (KLASIFIKASI) PANGKAT ASN
st.subheader('Probabilitas Hasil Prediksi (Klasifikasi) Pangkat ASN Kabupaten Sukoharjo')
st.write(prediksi_proba)

# MENAMPILKAN HASIL PREDIKSI (KLASIFIKASI) PANGKAT ASN
st.subheader('Hasil Prediksi (Klasifikasi) Pangkat ASN Kabupaten Sukoharjo')
result_df = pd.DataFrame({'Kode Pangkat': np.arange(len(prediksi)), 'Jenis Pangkat': jenis_pangkat[prediksi]})
st.table(result_df.set_index('Kode Pangkat'))
