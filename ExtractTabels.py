from pathlib import Path
from llama_index.readers.pdf_table import PDFTableReader
import camelot
import numpy as np

class PDFTableReaderCombined(PDFTableReader):
    def load_data(self, file: Path, pages: str = "all", extra_info=None):
        results = []

        # 1) LATTICE (ruled tables)
        lattice_tables = camelot.read_pdf(
            str(file), pages=pages, flavor="lattice",
            line_scale=40,
            strip_text="\n"
        )
        for table in lattice_tables:
            if self._is_valid_table(table):  
                doc = self._dataframe_to_document(df=table.df, extra_info=extra_info)
                results.append(doc)

        # 2) STREAM (whitespace heuristic)
        stream_tables = camelot.read_pdf(
            str(file), pages=pages, flavor="stream",
            edge_tol=50,
            strip_text="\n"
        )
        for table in stream_tables:
            if self._is_valid_table(table):  
                doc = self._dataframe_to_document(df=table.df, extra_info=extra_info)
                results.append(doc)

        return results

    def _is_valid_table(self, table) -> bool:
        """
        Rejects empty, low-density, or paragraph-like 'tables'.
        """
        df = table.df.fillna("").astype(str)

        # Basic shape
        n_rows, n_cols = df.shape
        if n_rows < 2 or n_cols < 2:
            return False

        # Density: proportion of non-empty cells
        non_empty = (df.applymap(lambda x: x.strip() != "")).to_numpy()
        non_empty_ratio = non_empty.mean()
        if non_empty_ratio < 0.4:   # skip sparse grids
            return False

        # Reject paragraph-like tables:
        # 1) Only one long column
        if n_cols == 1:
            avg_len = df[0].str.len().mean()
            if avg_len > 40:  # likely continuous text, not tabular
                return False

        # 2) Very uneven row/col ratio (too wide vs too tall)
        if n_cols > 12 and n_rows < 3:
            return False
        if n_rows > 200:
            return False

        # 3) Too much plain running text (spaces but no delimiters)
        text_join = " ".join(df.to_numpy().flatten())
        if len(text_join.split()) / max(1, (n_rows * n_cols)) > 8:
            # average >8 words per cell → probably a paragraph
            return False

        return True



# --- Main ---
pdf_file = Path("ResearchPapers/s5.pdf")
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

reader = PDFTableReaderCombined()
print(f"➡️ Processing {pdf_file.name} ...")

documents = reader.load_data(file=pdf_file, pages="all")

out_file = output_dir / f"{pdf_file.stem}.txt"
with out_file.open("w", encoding="utf-8") as f:
    f.write(f"Extracted {len(documents)} filtered tables from {pdf_file.name}\n\n")
    for i, doc in enumerate(documents, start=1):
        f.write(f"📊 Table {i}:\n")
        f.write((doc.text or "").rstrip() + "\n")
        f.write("-" * 80 + "\n")

print(f"  Saved results to {out_file}")
print("Done.")

