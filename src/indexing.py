import pathlib
import warnings


# je suis
class Indexer:
    def __init__(self, data_path: str | pathlib.Path):
        self.data_path = pathlib.Path(data_path)
        self.documents = dict(str, str)

    def extract_corpus(self):
        files = self.data_path.rglob("*")
        valid_files = ['.py', '.md', '.txt', '.rst']
        files = [
            file for ext in valid_files for file in files if file.endwith(ext)
            ]
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    source = f.read()
            except UnicodeDecodeError():
                warnings.warn("This is not a python file or markdown")
                continue
            if file.endwith('.py'):
                file_type = 'code'
            else:
                file_type = 'text'
            self.documents[file.relative_to(self.data_path)] = tuple(
                source, file_type)
