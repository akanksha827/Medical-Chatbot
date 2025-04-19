import json

def convert_ipynb_to_txt(ipynb_file, txt_file):
    try:
        # Open and read the .ipynb file
        with open(ipynb_file, 'r', encoding='utf-8') as file:
            notebook_data = json.load(file)

        # Open a .txt file to write the content
        with open(txt_file, 'w', encoding='utf-8') as output_file:
            for cell in notebook_data.get('cells', []):
                # Extract code or markdown content from each cell
                if cell.get('cell_type') in ['code', 'markdown']:
                    content = ''.join(cell.get('source', []))
                    output_file.write(content + '\n')
                    output_file.write('\n' + ('-' * 80) + '\n')  # Separator between cells

        print(f"Conversion successful! The content has been saved to {txt_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
ipynb_file = 'trails.ipynb'  # Replace with your .ipynb file path
txt_file = 'output.txt'  # Replace with your desired output .txt file path
convert_ipynb_to_txt(ipynb_file, txt_file)
