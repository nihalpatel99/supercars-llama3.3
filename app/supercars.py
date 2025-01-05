import pandas as pd
import chromadb
import uuid


class Supercars:
    def __init__(self, file_path="supercars.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="supercars")

    def load_supercars(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(documents=row["model"],
                       metadatas={"model" : row["model"],"engine": row["engine"],"top_speed": row["top_speed"]},
                       ids=[str(uuid.uuid4())])
                
    def query_cars(self, cars):
        return self.collection.query(query_texts=cars, n_results=2).get('metadatas', [])