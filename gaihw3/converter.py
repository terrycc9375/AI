import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF

def main():
    csv_path = "dev.csv"
    input_dir = Path("paper_evidence/dev")
    output_dir = Path("eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        paper_id = str(row["paper_id"])
        pdf_path = input_dir / f"{paper_id}.pdf"
        save_path = output_dir / f"{paper_id}.md"

        if not pdf_path.exists():
            continue

        if save_path.exists(): # 避免重複處理
            continue

        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                blocks = page.get_text("blocks")
                for block in blocks:
                    full_text += block[4] + "\n"
                full_text += "\n---Page Break---\n"
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            doc.close()
                
        except Exception as e:
            print(f"處理 {paper_id} 時發生錯誤: {e}")

if __name__ == "__main__":
    main()
