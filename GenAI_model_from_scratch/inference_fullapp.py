"""
Streamlit App for Medical Report Generation from CXR Images
------------------------------------------------------------
Author: [Your Name]
Date: [Current Date]

This app performs inference only. It loads the saved model ("cxr_report_generator.pth")
and vocabulary ("vocab.pkl"), generates a medical report from an uploaded chest X‑ray image,
augments the report using Gemini 2.0 Flash via the google-genai SDK, and creates a modern,
structured PDF for download.
"""

import os
import nest_asyncio
nest_asyncio.apply()  # Patch asyncio for Streamlit
import re
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet34, ResNet34_Weights
import pickle
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF
from datetime import datetime
from google import genai
from google.genai import types  # (Optional, if you need type definitions)

# -------------------------------
# Vocabulary Helper Class
# -------------------------------
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.count = 4

    def add_sentence(self, sentence):
        for word in sentence.split():
            if word not in self.word2idx:
                self.word2idx[word] = self.count
                self.idx2word[self.count] = word
                self.count += 1

    def numericalize(self, sentence, max_len=50):
        tokens = sentence.split()
        tokens = ["<start>"] + tokens + ["<end>"]
        token_ids = [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]
        if len(token_ids) < max_len:
            token_ids += [self.word2idx["<pad>"]] * (max_len - len(token_ids))
        else:
            token_ids = token_ids[:max_len]
        return token_ids

# -------------------------------
# Encoder-Decoder Model for Report Generation
# -------------------------------
class CXRReportGenerator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CXRReportGenerator, self).__init__()
        # Using ResNet-34 as encoder
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Output: (batch, 512, 1, 1)
        self.fc = nn.Linear(512, hidden_size)  # Map features to hidden size
        # Decoder
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, images, captions):
        features = self.encoder(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        embeddings = self.embed(captions)
        h0 = features.unsqueeze(0)
        c0 = torch.zeros_like(h0)
        outputs, _ = self.lstm(embeddings, (h0, c0))
        outputs = self.fc_out(outputs)
        return outputs

def load_model(model_path, embed_size, hidden_size, vocab_size, num_layers=1, device='cpu'):
    model = CXRReportGenerator(embed_size, hidden_size, vocab_size, num_layers)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    new_state_dict = {k.replace("module.", "") if k.startswith("module.") else k: v 
                      for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def generate_report(model, image_path, vocab, max_len=50, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    start_token = vocab.word2idx["<start>"]
    end_token = vocab.word2idx["<end>"]
    caption = torch.tensor([[start_token]], dtype=torch.long).to(device)
    generated_tokens = []
    with torch.no_grad():
        for i in range(max_len):
            outputs = model(image_tensor, caption)
            last_output = outputs[:, -1, :]
            predicted_token = last_output.argmax(dim=1).item()
            if predicted_token == end_token:
                break
            generated_tokens.append(predicted_token)
            caption = torch.cat([caption, torch.tensor([[predicted_token]], dtype=torch.long).to(device)], dim=1)
    generated_words = [vocab.idx2word.get(token, "<unk>") for token in generated_tokens]
    report = ' '.join(generated_words)
    return report, image

# -------------------------------
# Gemini 2.0 Flash API Call using google-genai SDK
# -------------------------------
def get_gemini_content(prompt):
    # Initialize the client with your API key (set as GEMINI_API_KEY env variable)
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY"))
    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents=prompt
    )
    return response.text

# -------------------------------
# Create PDF Function
# -------------------------------\
def clean_markdown(text):
    """
    Remove markdown formatting: leading hash symbols and bold markers.
    """
    # Remove markdown headings (hash symbols at the start of lines)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    # Remove bold markdown markers
    text = text.replace("**", "")
    return text
def create_pdf(report, additional_content, image, output_path="Medical_Report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # ----------------------------
    # Page 1: Structured Text Report
    # ----------------------------
    pdf.add_page()
    
    # Title and Date
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Chest X-Ray Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    pdf.ln(5)
    
    # Patient and Study Information
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Patient Name: [Patient Name]", ln=True)
    pdf.cell(0, 10, "Patient MRN: [Patient MRN]", ln=True)
    pdf.cell(0, 10, "Date of Study: 2023-10-27", ln=True)
    pdf.cell(0, 10, "Study Type: Chest Radiograph (PA and Lateral)", ln=True)
    pdf.cell(0, 10, "Referring Physician: [Referring Physician Name]", ln=True)
    pdf.ln(5)
    
    # Clinical Indication (Placeholder)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Clinical Indication:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "[Clinical Indication details here]")
    pdf.ln(5)
    
    # Findings (Model-generated report)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Findings:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, report)
    pdf.ln(5)
    
    # Impression (Additional content with markdown cleaned)
    if additional_content and "Error" not in additional_content:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Impression:", ln=True)
        pdf.set_font("Arial", size=12)
        cleaned_impression = clean_markdown(additional_content)
        pdf.multi_cell(0, 10, cleaned_impression)
        pdf.ln(5)
    
    # ----------------------------
    # Page 2: X-Ray Image
    # ----------------------------
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Attached X-Ray Image:", ln=True)
    pdf.ln(5)
    
    temp_img_path = "temp_img_for_pdf.jpg"
    image.save(temp_img_path)
    pdf.image(temp_img_path, x=15, w=pdf.w - 30)
    os.remove(temp_img_path)
    
    pdf.output(output_path)
    return output_path

# -------------------------------
# Streamlit Inference App (Inference Only)
# -------------------------------
def main():
    st.title("CXR Medical Report Generation")
    st.write("Upload a chest X-ray image to generate a detailed, structured medical report.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Using device: {device}")

    # Load saved vocabulary and model (assumes vocab.pkl and cxr_report_generator.pth exist)
    try:
        with st.spinner('Loading vocabulary...'):
            with open('models/vocab.pkl', 'rb') as f:
                vocab = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading vocabulary: {e}")
        return
    vocab_size = vocab.count
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    model_path = "models/cxr_report_generator.pth"
    try:
        with st.spinner('Loading model...'):
            model = load_model(model_path, embed_size, hidden_size, vocab_size, num_layers, device)
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Upload image
    uploaded_file = st.file_uploader("Upload a CXR Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)
        temp_image_path = "temp_uploaded_image.jpg"
        image.save(temp_image_path)
        
        # Generate report from the model
        with st.spinner('Generating medical report...'):
            generated_report, processed_image = generate_report(model, temp_image_path, vocab, max_len=100, device=device)
        st.subheader("Generated Medical Report by my own GenAI model")
        st.write(generated_report)
        
        # Generate additional content using Gemini 2.0 Flash
        gemini_prompt = f"Generate a detailed, structured medical report for a chest X-ray with the following summary: {generated_report}"
        with st.spinner('Augmenting report with Gemini 2.0 Flash...'):
            additional_content = get_gemini_content(gemini_prompt)
        st.subheader("Additional Findings (Gemini 2.0 Flash)")
        st.write(additional_content)
        
        # Visualize the image with overlay of report (for preview)
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.imshow(processed_image)
        ax.axis('off')
        ax.set_title("X-Ray Image", fontsize=20, fontweight='bold', pad=20)
        plt.subplots_adjust(bottom=0.3)
        fig.text(0.5, 0.05, "Report Summary:\n" + generated_report + "\n\nAdditional Findings:\n" + additional_content,
                 ha='center', fontsize=14, wrap=True)
        st.pyplot(fig)
        
        # Option to download the generated report as a PDF in modern medical document format
        if st.button("Download Medical Report as PDF"):
            with st.spinner('Creating PDF...'):
                pdf_path = create_pdf(generated_report, additional_content, processed_image)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="Medical_Report.pdf", mime="application/pdf")
    else:
        st.info("Please upload an X-ray image to generate a report.")

if __name__ == '__main__':
    main()
