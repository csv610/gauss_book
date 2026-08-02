"""Fix nested index entries and expand chapters."""

import os
import re

chapters_dir = "/Users/csv610/Projects/MyBooks/Gauss/chapters"

# First, fix nested index entries
for filename in os.listdir(chapters_dir):
    if not filename.endswith('.tex') or filename.startswith('appendix'):
        continue
    
    filepath = os.path.join(chapters_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove nested \index{} commands
    # Pattern: \index{term\index{term}}
    content = re.sub(r'\\index{([^}]*)\\index{[^}]*}}', r'\\index{\1}', content)
    # Remove duplicate \index{} entries
    content = re.sub(r'\\index{([^}]*)} \\index{\\1}', r'\\index{\1}', content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Fixed {filename}")

print("Done!")
