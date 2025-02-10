from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

def Llama_document_parser() -> pd.DataFrame:
    pdf_files = ["..\Codes\The New Complete Book of Foos.pdf"]
    # set up parser
    parser = LlamaParse(result_type="text", api_key="API KEY")

    file_extractor = {".pdf": parser}

    data_for_parse = SimpleDirectoryReader(input_files=pdf_files, file_extractor=file_extractor)

    documents = data_for_parse.load_data()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=64,
        length_function=len,
        is_separator_regex=False,
    )

    documents_list = []
    page_number = 0
    last_doc = None
    for doc in documents:
        if last_doc is None or last_doc != doc.metadata["file_name"]:
            page_number = 1
            last_doc = doc.metadata["file_name"]
        else:
            page_number += 1

        texts = text_splitter.split_text(doc.text)
        for text in texts:
            item = {}
            item["id_"] = doc.id_
            item["text"] = text
            item["metadata_file_name"] = doc.metadata["file_name"]
            item["metadata_creation_date"] = doc.metadata["creation_date"]
            item["metadata_pagenumber"] = page_number
            documents_list.append(item)

    df = pd.DataFrame(documents_list)

    return df
    