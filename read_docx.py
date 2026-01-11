from docx import Document
import os

DOC_PATH = r"c:\Users\NQT\Desktop\OneDrive\02_ACADEMICS\01_HK1\ComputerVision\00_Thesis\EE5205_MatchingTool_Gr2\EE5205_ProjectVision.docx"
import sys
sys.stdout.reconfigure(encoding='utf-8')

if not os.path.exists(DOC_PATH):
    print("File not found")
else:
    doc = Document(DOC_PATH)
    print("--- START OF DOCUMENT ---")
    for para in doc.paragraphs:
        if para.text.strip():
            print(para.text)
    print("--- END OF DOCUMENT ---")
