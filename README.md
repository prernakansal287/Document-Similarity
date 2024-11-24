
# Document Similarity

This project compares PDF invoices based on both textual and visual similarities. It uses multiple techniques to evaluate the similarity, including TF-IDF and cosine similarity for text comparison, Jaccard similarity for word set comparison, and Structural Similarity Index (SSIM) for image comparison.

## Features

- **Text Extraction from PDFs**: Extracts text from PDF invoices using the `PyMuPDF` and `pdfplumber` libraries.
- **Text Preprocessing**: Preprocesses the text by converting it to lowercase, removing special characters, tokenizing, removing stopwords, and stemming the words.
- **Textual Similarity Measures**: Uses TF-IDF and cosine similarity, as well as Jaccard similarity, to compare the text content of invoices.
- **Image Similarity**: Converts PDF pages to images and uses SSIM to compare the visual similarity between two PDF pages.
- **Invoice Comparison**: Compares a given invoice to a database of invoices and finds the most similar invoice based on both text and image similarities.

## Requirements

- `fitz` (PyMuPDF)
- `pdfplumber`
- `nltk`
- `scikit-learn`
- `scikit-image`
- `opencv-python`
- `pytesseract`
- `Pillow`

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## How it Works

1. **Text Extraction**: Text is extracted from PDF files.
2. **Text Preprocessing**: The extracted text is preprocessed for similarity comparison.
3. **Similarity Calculation**: Text similarities are calculated using TF-IDF, cosine similarity, and Jaccard index.
4. **Image Similarity**: Pages are saved as images and compared using SSIM.
5. **Comparison**: The program compares an input invoice with a set of invoices in a database and returns the most similar invoice based on both textual and visual similarity.

## Example Usage

To compare an invoice from the `testDatabase` with invoices from the `trainDatabase`, simply run the script. It will print the most similar invoice and the similarity score for each invoice in the test database.

```python
trainDatabase = [
    {'path': '2024.03.15_0954.pdf'},
    {'path': '2024.03.15_1145.pdf'},
    {'path': 'Faller_8.pdf'},
    {'path': 'invoice_77073.pdf'},
    {'path': 'invoice_102856.pdf'},
]

testDatabase = [
    {'path': 'invoice_102857.pdf'},
    {'path': 'invoice_77098.pdf'},
]

for x in testDatabase: 
    input_invoice_path = x['path']
    best_match, similarity_score = compare_invoices(input_invoice_path, trainDatabase)
    print(f"Best Match: {best_match['path']}, Similarity Score: {similarity_score}")
```
