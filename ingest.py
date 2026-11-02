"""Load files into the Deep OQ database.

  python ingest.py path/to/folder_or_file [more paths...]

Supports .txt .md .py .js .java .cpp .html .json .csv and .pdf (needs: pip install pypdf).
Put your coding docs, notes, news, books, GK files in a folder and ingest them.
Safe to run again on new files; each file is stored by its path.
"""
import os
import sys
import kb

TEXT_EXT = {".txt", ".md", ".py", ".js", ".ts", ".java", ".cpp", ".c", ".html",
            ".css", ".json", ".csv", ".kt", ".go", ".rs", ".sql"}


def read_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    if ext in TEXT_EXT:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return None


def main(paths):
    total = 0
    for p in paths:
        files = []
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in names]
        else:
            files = [p]
        for f in files:
            try:
                text = read_file(f)
            except Exception as e:
                print("skip", f, e)
                continue
            if not text or not text.strip():
                continue
            n = kb.add_text(text, title=os.path.basename(f), source=f)
            total += n
            print(f"{f}: {n} chunks")
    print("done.", kb.stats(), "added", total)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        main(sys.argv[1:])
