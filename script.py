import fitz  # PyMuPDF
import pdfplumber
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import pytesseract
from pytesseract import Output
from PIL import Image
import io

# Ensure you have downloaded the required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    all_text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        page_text = page.get_text()
        all_text += page_text
    document.close()
    return all_text

# Preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    return words

# Compute TF-IDF and cosine similarity
def compute_tfidf_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(text1), ' '.join(text2)])
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_matrix[0][0]

# Compute Jaccard similarity
def compute_jaccard_similarity(text1, text2):
    set1 = set(text1)
    set2 = set(text2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(intersection) / len(union)

# Analyze PDF layout
def analyze_pdf_layout(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        headers = []
        footers = []
        tables = []
        for page in pdf.pages:
            tables.extend(page.extract_tables())
            text = page.extract_text()
            lines = text.split('\n')
            headers.append(lines[0])
            footers.append(lines[-1])
        return headers, footers, tables

# Save PDF page as an image
def save_page_as_image(pdf_path, page_num, image_path):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap()
    pix.save(image_path)
    doc.close()

# Compute image similarity
def compute_image_similarity(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise ValueError("One or both images not found or unable to read.")
    img1 = cv2.resize(img1, (600, 800))
    img2 = cv2.resize(img2, (600, 800))
    ssim_index, _ = ssim(img1, img2, full=True)
    return ssim_index

# Compare input invoice with database
def compare_invoices(input_pdf_path, database):
    input_text = extract_text_from_pdf(input_pdf_path)
    preprocessed_input_text = preprocess_text(input_text)
    input_headers, input_footers, input_tables = analyze_pdf_layout(input_pdf_path)
    save_page_as_image(input_pdf_path, 0, 'input_img.png')
    best_match = None
    highest_similarity = 0.0
    for invoice in database:
        db_text = extract_text_from_pdf(invoice['path'])
        preprocessed_db_text = preprocess_text(db_text)
        cosine_sim = compute_tfidf_cosine_similarity(preprocessed_input_text, preprocessed_db_text)
        jaccard_sim = compute_jaccard_similarity(preprocessed_input_text, preprocessed_db_text)
        text_similarity = (cosine_sim + jaccard_sim) / 2
        db_headers, db_footers, db_tables = analyze_pdf_layout(invoice['path'])
        save_page_as_image(invoice['path'], 0, 'db_img.png')
        structural_similarity = compute_image_similarity('input_img.png', 'db_img.png')
        overall_similarity = (text_similarity + structural_similarity) / 2
        if overall_similarity > highest_similarity:
            highest_similarity = overall_similarity
            best_match = invoice
    return best_match, highest_similarity

# Example usage
trainDatabase = [
    {'path': '2024.03.15_0954.pdf'},
    {'path': '2024.03.15_1145.pdf'},
    {'path': 'Faller_8.pdf'},
    {'path': 'invoice_77073.pdf'},
    {'path': 'invoice_102856.pdf'},
]

testDatabase= [
    {'path': 'invoice_102857.pdf'},
    {'path': 'invoice_77098.pdf'},
]
i=1
for x in testDatabase: 
    print('Printing similarity of path number: ',i)
    i+=1
    input_invoice_path = x['path']
    best_match, similarity_score = compare_invoices(input_invoice_path, testDatabase)
    print(f"Best Match: {best_match['path']}, Similarity Score: {similarity_score}")