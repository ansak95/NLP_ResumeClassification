import PyPDF2
import io
import pytesseract
from PIL import Image
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.

    Args:
    pdf_path: Path to the PDF file.

    Returns:
    Extracted text as a string.
    """
    text = "" # Initialize an empty string to store the extracted text
    try:
        with open(pdf_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)  # Create a PyPDF2 PdfFileReader object

            # Iterate through each page in the PDF
            for page_num in range(len(pdf.pages)):  # Use len(pdf.pages) to get the number of pages
                page = pdf.pages[page_num]  # Use pdf.pages[page_num] to get the specific page
                text_content = page.extract_text()  # Extract text from the page

                if text_content.strip():  # If there is non-empty text directly extracted from the page
                    text += text_content  # Append it to the text variable
                else: # If no text is directly extracted (e.g., scanned image or non-text content)
                    print("Tesseract used")
                    try:
                        # Convert the PDF page to an image
                        image = page.toImage()
                        image_bytes = io.BytesIO()
                        image.save(image_bytes, format='JPEG')
                        image_bytes = image_bytes.getvalue()
                        
                        # Use pytesseract to extract text from the image
                        text += pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)))
                    except Exception as e:
                        print(f"Error extracting page {page_num}: {e}")
    except Exception as e:
        print(f"Error opening PDF: {e}")
    return text


# Function to preprocess and clean the text
def preprocess_text(text):
    """
    Preprocess and clean the text.

    Args:
    text: Text to be processed.

    Returns:
    Processed and cleaned text.
    """
    # Remove special characters, punctuation, and extra whitespaces
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower() # Convert text to lowercase
    
    # Tokenize the text (split it into words)
    words = text.split()
    # Remove stopwords (common words like 'the', 'and', 'in' that don't carry much meaning)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize words (reduce them to their base form)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the cleaned and lemmatized words back into a single string
    return ' '.join(words)


# Function to remove custom stopwords from a text
def remove_custom_stopwords(text, custom_stopwords):
    """
    Remove custom stopwords from the text.

    Args:
    text: Text to be processed.
    custom_stopwords: List of custom stopwords to remove.

    Returns:
    Text with custom stopwords removed.
    """
    words = text.split()
    filtered_words = [word for word in words if word not in custom_stopwords]
    
    # Tokenize the text into words
    words = text.split()
    # Remove custom stopwords
    words = [word for word in words if word not in custom_stopwords]
    # Remove words shorter than 3 characters
    filtered_words = [word for word in words if len(word) > 2]
    
    # Join the filtered words back into a single string    
    return ' '.join(filtered_words)