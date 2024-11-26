import PyPDF2

def remove_pages(input_pdf_path, output_pdf_path, start_page, end_page):
    # Open the original PDF
    with open(input_pdf_path, "rb") as input_pdf:
        reader = PyPDF2.PdfReader(input_pdf)
        
        # Create a writer object to save the modified PDF
        writer = PyPDF2.PdfWriter()

        # Loop through all the pages in the original PDF
        for i in range(len(reader.pages)):
            # Skip the pages in the specified range (1-indexed range)
            if i < start_page - 1 or i > end_page - 1:
                writer.add_page(reader.pages[i])

        # Write the new PDF to a file
        with open(output_pdf_path, "wb") as output_pdf:
            writer.write(output_pdf)

# Example usage:
remove_pages("output.pdf", "input.pdf", start_page=1, end_page=11)
