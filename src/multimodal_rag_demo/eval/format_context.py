def format_text_chunks(matches: dict) -> str:
    if not matches:
        return "(no text chunks retrieved)"
    return "\n".join(
        f"- [{entry['file_name']} p.{entry['page_num']}] {entry['chunk_text']}"
        for entry in matches.values()
    )


def format_image_descriptions(matches: dict) -> str:
    if not matches:
        return "(no images retrieved)"
    return "\n".join(
        f"- [{entry['file_name']} p.{entry['page_num']}] {entry['image_description']}"
        for entry in matches.values()
    )
