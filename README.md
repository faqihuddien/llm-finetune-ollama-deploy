# Fine-Tuning LLM untuk Domain-Specific QA + Deployment via Ollama

Fine-tuning model bahasa kecil (Qwen2.5-1.5B-Instruct) menggunakan LoRA untuk menjawab pertanyaan seputar sistem traffic monitoring berbasis computer vision, lalu men-deploy hasilnya secara lokal menggunakan Ollama.

## Latar Belakang

Project ini dibuat sebagai studi kasus end-to-end pipeline LLM: mulai dari penyiapan dataset domain-specific, fine-tuning dengan parameter-efficient method (LoRA), konversi model ke format quantized (GGUF), hingga deployment lokal yang siap diintegrasikan ke aplikasi.

Domain yang dipilih — traffic monitoring — terhubung dengan project computer vision penulis sebelumnya (deteksi kendaraan dengan YOLOv11, tracking dengan DeepSORT, prediksi kepadatan lalu lintas dengan LSTM, dan estimasi kecepatan berbasis homography transformation). Model hasil fine-tuning ini berperan sebagai asisten QA yang dapat menjelaskan konsep-konsep teknis dari sistem tersebut.

## Fitur

- Fine-tuning LLM dengan **LoRA** (parameter-efficient, hemat resource, bisa jalan di GPU T4 gratis Colab)
- Training dipercepat menggunakan **Unsloth** (~2x lebih cepat, memori lebih hemat dibanding implementasi standar)
- Dataset custom berformat Alpaca (`instruction` / `output`) di domain traffic monitoring
- Export otomatis ke format **GGUF** dengan quantization Q4_K_M (ukuran kecil, kualitas inferensi tetap baik)
- Deployment lokal menggunakan **Ollama**, siap dipanggil lewat REST API untuk integrasi aplikasi

## Tech Stack

| Komponen | Tools |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct |
| Fine-tuning | LoRA via Unsloth + TRL (SFTTrainer) |
| Environment training | Google Colab (GPU T4) |
| Format model | GGUF (Q4_K_M quantization) |
| Deployment | Ollama |

## Struktur Project

```
.
├── notebooks/
│   └── finetune_llm_ollama.ipynb   # Notebook fine-tuning end-to-end (Colab)
├── dataset/
│   └── traffic_monitoring_qa.jsonl # Dataset instruction-response
├── model-gguf/
│   ├── Qwen2.5-1.5B-Instruct.Q4_K_M.gguf
│   └── Modelfile                   # Konfigurasi Ollama
└── README.md
```

## Dataset

Dataset berisi 62 pasangan instruksi-jawaban seputar konsep computer vision untuk traffic monitoring, contoh:

```json
{"instruction": "Apa itu sistem traffic monitoring berbasis AI?", "output": "Sistem traffic monitoring berbasis AI adalah sistem yang menggunakan kecerdasan buatan untuk mendeteksi, melacak, menghitung, dan menganalisis kondisi lalu lintas dari data seperti video CCTV secara otomatis."}
```

Topik yang dicakup: deteksi objek (YOLO), object tracking (DeepSORT), prediksi kepadatan (LSTM), dan estimasi kecepatan berbasis homography.

## Proses Fine-Tuning

1. Load base model dalam mode 4-bit quantized menggunakan Unsloth (`FastLanguageModel`)
2. Tambahkan LoRA adapter pada attention & MLP layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
3. Format dataset ke Alpaca prompt template
4. Training dengan `SFTTrainer` (TRL) — 60 steps, learning rate 2e-4, batch size efektif 8 (via gradient accumulation)
5. Export model ke GGUF dengan quantization Q4_K_M
6. Generate `Modelfile` otomatis untuk Ollama

Detail lengkap ada di [`notebooks/finetune_llm_ollama.ipynb`](notebooks/finetune_llm_ollama.ipynb).

## Model

File model hasil fine-tuning (`.gguf`, ~962MB) tidak disertakan di repo ini karena keterbatasan ukuran GitHub. Download di sini:

📦 [Download model GGUF (Google Drive)](https://drive.google.com/file/d/1i0ge-zdaFlXCvstCc2fppjm3-12ROtfK/view?usp=drive_link)

Setelah download, taruh file di folder `model-gguf/` (sejajar dengan `Modelfile`).

## Cara Menjalankan (Deployment Lokal)

**Prasyarat:** [Ollama](https://ollama.com/download) sudah terinstall.

1. Clone repo ini dan masuk ke folder model:
   ```bash
   cd model-gguf
   ```

2. Build model di Ollama dari `Modelfile`:
   ```bash
   ollama create traffic-qa-model -f Modelfile
   ```

3. Jalankan model:
   ```bash
   ollama run traffic-qa-model
   ```

## Contoh Hasil

| Pertanyaan | Jawaban Model |
|---|---|
| Apa itu sistem traffic monitoring? | Sistem traffic monitoring adalah teknologi pengamatan kecepatan, posisi kendaraan, dan kondisi jalan secara
real-time menggunakan CCTV atau kamera video. |
| Bagaimana cara kerja YOLO untuk deteksi kendaraan? | Yolo (You Only Look Once) menggunakan teknik sliding window dan bounding box untuk mempertahankan kelas,
confidensi, dan lokasi objek sekaligus dalam satu tahap prediksi. |

## Pengembangan Lanjutan

- [ ] Perbandingan throughput/latency deployment Ollama vs vLLM
- [ ] Perluasan dataset (100+ pasang instruction-response) untuk kualitas jawaban yang lebih konsisten
- [ ] Evaluasi kuantitatif (BLEU/ROUGE atau human evaluation) terhadap jawaban model base vs hasil fine-tuning
- [ ] Integrasi ke aplikasi via Ollama REST API

## Author

Faza Muhammad Faqihuddien
[Linkedin](https://linkedin.com/in/faza-faqihuddien)
[CV](https://drive.google.com/file/d/1oPBo-ER70uymrJTI4M1C16xtmcDoMROk/view?usp=sharing)
