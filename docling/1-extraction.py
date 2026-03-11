from docling.document_converter import DocumentConverter

converter = DocumentConverter()

#result= converter.convert("./docling/2408.09869v5.pdf")
result = converter.convert("https://arxiv.org/abs/2408.09869")

document = result.document

markdown_output = document.export_to_markdown()
print(markdown_output)  

# Convertendo pagina HTML
result = converter.convert("https://docling-project.github.io/docling/")
document = result.document
markdown_output = document.export_to_markdown()
print(markdown_output)  

