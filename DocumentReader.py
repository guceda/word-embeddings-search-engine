import os
import glob

class DocumentReader(object):
    def __init__(self, path):
        self.path = path

    @property
    def corpus(self):
        documents = []
        glob_path = os.path.join(self.path, "**")
        for document_path in glob.glob(glob_path, recursive=True):
            if os.path.isfile(document_path):
                with open(document_path, 'r', encoding='ISO-8859-1') as f:
                    documents.append(f.read())
        return documents