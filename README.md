# Image-Retrieval-System-Using-CLIP-and-FAISS

## Overview
This project is an **image retrieval system** that takes a **text prompt** and retrieves the most relevant images from a dataset. It utilizes **CLIP-based embeddings** for encoding images and **FAISS indexing** for fast similarity search. The interface is built using **Gradio** for easy user interaction.

---

## Workflow

1. **Dataset Loading:**  
   - The dataset is downloaded and loaded using `datasets.load_dataset("imagefolder")`.

2. **Image Encoding Using CLIP:**  
   - The **CLIP-ViT-B-32** model encodes images into embeddings.
   - Each image's embedding is stored in the dataset.

3. **Adding FAISS Indexing:**  
   - A FAISS index is added to enable fast similarity search on embeddings.

4. **Retrieving Similar Images:**  
   - A user inputs a text prompt.
   - The prompt is converted into an embedding.
   - FAISS finds the **top-k similar images** based on similarity.

5. **Displaying Retrieved Images:**  
   - The retrieved images are displayed using `matplotlib`.

6. **Gradio Web Interface:**  
   - A Gradio web app is provided for user interaction.

---

## Tech Stack

- **Data Processing & Loading:**
  - `datasets` (Hugging Face) - Loads image dataset
  - `pandas` (optional) - Data transformation

- **Embedding Model:**
  - `sentence-transformers` - Uses **CLIP (ViT-B-32)** for embedding

- **Image Retrieval & Indexing:**
  - `FAISS` (Facebook AI Similarity Search) - Efficient nearest neighbor search

- **Visualization:**
  - `matplotlib.pyplot` - Displays retrieved images

- **Web Interface:**
  - `Gradio` - Builds an interactive UI for searching images

---

## Installation & Setup

### **1. Clone the repository**
```sh
 git clone https://github.com/your-repo/image-retrieval.git
 cd image-retrieval
```

### **2. Install dependencies**
```sh
pip install -r requirements.txt
```

### **3. Run the application**
```sh
python script.py
```
