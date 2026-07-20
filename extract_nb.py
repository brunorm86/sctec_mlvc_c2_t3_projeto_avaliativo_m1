import json
with open('projeto_credito.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
with open('notebook_extracted.py', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        out.write(f"# CELL {i} ({cell['cell_type']})\n")
        if isinstance(cell.get('source'), list):
            for line in cell.get('source', []):
                out.write(line)
        else:
            out.write(cell.get('source', ''))
        out.write('\n' + '-'*40 + '\n')
