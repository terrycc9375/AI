import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

def main():
    csv_path = "train.csv"
    input_dir = Path("paper_evidence/train")
    output_dir = Path("train")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "output_format": "markdown",
        "disable_image_extraction": True,  # 禁用圖片提取
    }
    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service()
    )
    df = pd.read_csv(csv_path)
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        paper_id = str(row["paper_id"])
        pdf_path = input_dir / f"{paper_id}.pdf"
        save_path = output_dir / f"{paper_id}.md"

        if not pdf_path.exists():
            print(f"警告: 找不到檔案 {pdf_path}")
            continue

        try:
            rendered = converter(str(pdf_path))
            full_text, _, _ = text_from_rendered(rendered)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(full_text)
                
        except Exception as e:
            print(f"處理 {paper_id} 時發生錯誤: {e}")

if __name__ == "__main__":
    main()
