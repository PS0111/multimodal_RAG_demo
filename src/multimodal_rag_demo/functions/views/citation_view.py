def format_text_citations(results: dict) -> list:
    return [
        {
            "cosine_score": float(entry["cosine_score"]),
            "file_name": entry["file_name"],
            "page_num": int(entry["page_num"]),
            "chunk_number": int(entry["chunk_number"]),
            "chunk_text": entry["chunk_text"],
        }
        for entry in results.values()
    ]


def format_image_citations(results: dict) -> list:
    return [
        {
            "cosine_score": float(entry["cosine_score"]),
            "file_name": entry["file_name"],
            "page_num": int(entry["page_num"]),
            "img_path": entry["img_path"],
            "image_description": entry["image_description"],
        }
        for entry in results.values()
    ]
