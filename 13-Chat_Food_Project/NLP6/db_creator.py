import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
import pandas as pd


def df_to_dict_batches(df: pd.DataFrame, batch_size: int = 128):
    """
    Yields data from a DataFrame in batches of dictionaries.
    Each batch is a list of dict, suitable for LanceDB ingestion.
    """
    for start_idx in range(0, len(df), batch_size):
        end_idx = start_idx + batch_size
        # Convert the batch of rows to a list of dict
        batch_dicts = df.iloc[start_idx:end_idx].to_dict(orient="records")
        yield batch_dicts


def df_to_dbtbl(db, df: pd.DataFrame):
    embedding_model = get_registry().get("sentence-transformers").create(name="BAAI/bge-small-en-v1.5", device="cpu")

    class ChunksOfData(LanceModel):
        id_: str
        text: str = embedding_model.SourceField()
        metadata_file_name: str
        metadata_creation_date: str
        metadata_pagenumber: int
        vector: Vector(embedding_model.ndims()) = embedding_model.VectorField()

    tbl = db.create_table(
        "embedded_chunks2",
        data=df_to_dict_batches(df, batch_size=10),
        schema=ChunksOfData,
    )
    # tbl = db.open_table("embedded_chunks")
    
    return tbl
