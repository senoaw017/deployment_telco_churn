# 🚀 Panduan Deployment Streamlit App

## 📁 File-file yang Sudah Disiapkan

Semua file sudah tersedia di **AI Drive: `/churn_app/`**

1. ✅ **app.py** - Main Streamlit application (13.9 KB)
2. ✅ **requirements.txt** - Dependencies list
3. ✅ **README.md** - Dokumentasi lengkap

## 📦 File Tambahan yang Anda Perlukan

Anda perlu menambahkan 2 file ini ke folder `/churn_app/`:

1. **churn_model.joblib** - Model yang sudah Anda train
2. **customerchurn.csv** - Dataset asli

## 🎯 Cara Deploy

### **Option 1: Local (Komputer Anda)**

```bash
# 1. Download semua file dari AI Drive /churn_app/
# 2. Pindahkan churn_model.joblib dan customerchurn.csv ke folder yang sama
# 3. Install dependencies
pip install -r requirements.txt

# 4. Run app
streamlit run app.py
```

App akan terbuka di: `http://localhost:8501`

---

### **Option 2: Deploy ke Streamlit Cloud (Gratis)**

#### Step 1: Persiapan
1. Upload semua file ke **GitHub repository**:
   ```
   your-repo/
   ├── app.py
   ├── requirements.txt
   ├── churn_model.joblib
   ├── customerchurn.csv
   └── README.md
   ```

#### Step 2: Deploy
1. Buka https://streamlit.io/cloud
2. Sign in dengan GitHub
3. Click "New app"
4. Pilih repository Anda
5. Main file: `app.py`
6. Click "Deploy"!

**URL**: `https://your-app-name.streamlit.app`

---

### **Option 3: Deploy ke Hugging Face Spaces (Gratis)**

#### Step 1: Create Space
1. Buka https://huggingface.co/spaces
2. Click "Create new Space"
3. Pilih **Streamlit** SDK
4. Beri nama space Anda

#### Step 2: Upload Files
Upload semua file:
- app.py
- requirements.txt  
- churn_model.joblib
- customerchurn.csv

#### Step 3: Auto Deploy
Hugging Face akan otomatis deploy!

**URL**: `https://huggingface.co/spaces/username/space-name`

---

## 🔧 Troubleshooting

### Error: "Model not found"
**Fix**: Pastikan `churn_model.joblib` ada di folder yang sama dengan `app.py`

### Error: "CSV not found"
**Fix**: Pastikan `customerchurn.csv` ada di folder yang sama

### Error: Module not found
**Fix**: Install ulang dependencies:
```bash
pip install -r requirements.txt
```

### Port sudah digunakan
**Fix**: Gunakan port lain:
```bash
streamlit run app.py --server.port 8502
```

---

## 📊 Testing

Setelah deploy, test dengan input:
- **High Risk Customer**: Month-to-month, Fiber optic, tenure < 12 bulan
- **Low Risk Customer**: Two year contract, DSL, tenure > 24 bulan

---

## 🎨 Customization

Edit `app.py` untuk:
- Ubah warna di CSS section (line 10-40)
- Tambah/kurangi fields
- Ubah threshold churn (default 50%)
- Custom recommendations

---

## 📝 Next Steps

1. Download file dari AI Drive `/churn_app/`
2. Tambahkan `churn_model.joblib` dan `customerchurn.csv`
3. Test local dulu dengan `streamlit run app.py`
4. Kalau OK, deploy ke cloud (Streamlit Cloud/Hugging Face)
5. Share URL ke team! 🎉

---

## 💡 Tips

- **Streamlit Cloud**: Best untuk internal team (gratis, private repo OK)
- **Hugging Face**: Best untuk public showcase (gratis, komunitas besar)
- **Local**: Best untuk testing/development

---

**Need Help?** Cek README.md untuk dokumentasi lengkap!
