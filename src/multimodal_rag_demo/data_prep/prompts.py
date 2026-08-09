IMAGE_DESCRIPTION_PROMPT = """You are analyzing an image extracted from an MBA course slide deck. This may be a business framework diagram (2x2 matrix, process flow, value chain, org chart), a data chart or graph, a photo, or a screenshot.

Your task is to produce a description that captures every piece of information a student would need to answer questions about this image later, without seeing it again.

Instructions:
- If it is a framework or matrix: identify the axes and their labels, every quadrant/category and its name, and any definitions or examples shown within each quadrant.
- If it is a chart or graph: identify the type of chart, the axes and their units, and the key values, trends, or comparisons shown.
- If it is a process/flow diagram: list each step in sequence and any labeled transitions or decision points.
- If it contains any text, numbers, or labels, transcribe them exactly rather than paraphrasing.
- Do not add interpretation, commentary, or business advice beyond what is visibly shown.
- If any part of the image is unclear or a label is unreadable, state "Unknown" or "Not visible" for that part rather than guessing.
"""
