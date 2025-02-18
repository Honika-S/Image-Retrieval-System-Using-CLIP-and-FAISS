import os
import matplotlib.pyplot as plt
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rich import inspect
import gradio as gr

# Define the paths
DATA_FILE_URL = "https://zenodo.org/record/6224034/files/embellishments_sample.zip?download=1"

# Global variables for dataset and model
dataset_with_embeddings = None
model = None

def load_and_prepare_dataset(data_file_url: str):
    """Load the dataset and prepare it."""
    dataset = load_dataset("imagefolder", data_files=data_file_url)
    return dataset

def encode_images_with_model(dataset, model_name='clip-ViT-B-32', batch_size=32):
    """Encode images using the specified model."""
    model = SentenceTransformer(model_name)
    dataset_with_embeddings = dataset.map(
        lambda example: {'embeddings': model.encode(example['image'], device='cpu')}, 
        batched=True, 
        batch_size=batch_size
    )
    return dataset_with_embeddings, model

def add_faiss_index(dataset):
    """Add a FAISS index to the dataset based on embeddings."""
    dataset['train'].add_faiss_index(column='embeddings')

def retrieve_examples(dataset, model, prompt, k=9):
    """Retrieve nearest examples based on a prompt."""
    encoded_prompt = model.encode(prompt)
    scores, retrieved_examples = dataset['train'].get_nearest_examples('embeddings', encoded_prompt, k=k)
    return retrieved_examples

def plot_retrieved_images(retrieved_examples, columns=3):
    """Plot retrieved images."""
    fig, axs = plt.subplots((len(retrieved_examples['image']) + columns - 1) // columns, columns, figsize=(20, 20))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        if i < len(retrieved_examples['image']):
            ax.imshow(retrieved_examples['image'][i])
            ax.axis('off')
        else:
            ax.axis('off')
    return fig

def preprocess_data():
    global dataset_with_embeddings, model
    # Load dataset
    print("Loading dataset...")
    dataset = load_and_prepare_dataset(DATA_FILE_URL)
    print("Dataset loaded.")

    # Encode images
    print("Encoding images...")
    dataset_with_embeddings, model = encode_images_with_model(dataset)
    print("Images encoded.")

    # Add FAISS index
    print("Adding FAISS index...")
    add_faiss_index(dataset_with_embeddings)
    print("FAISS index added.")

def gradio_main(prompt):
    # Retrieve examples
    print("Retrieving examples...")
    retrieved_examples = retrieve_examples(dataset_with_embeddings, model, prompt)
    print("Examples retrieved.")

    # Plot images
    print("Plotting images...")
    fig = plot_retrieved_images(retrieved_examples)
    print("Images plotted.")
    
    return fig

def main():
    preprocess_data()

    # Gradio Interface
    iface = gr.Interface(
        fn=gradio_main,
        inputs="text",
        outputs="plot",
        title="Image Retrieval Based on Prompt",
        description="Enter a prompt and get the most relevant images."
    )

    iface.launch()

if __name__ == "__main__":
    main()
