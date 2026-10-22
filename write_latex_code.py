# write_latex_code.py

from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool


@tool
def render_latex_code(latex_content: str) -> str:
    """Save a LaTeX document as a .tex file without PDF compilation.

    Args:
        latex_content: The complete LaTeX document content as a string.

    Returns:
        Absolute path to the generated LaTeX (.tex) file.
    """

    try:
        # Step 1: Create output directory
        output_dir = Path("output").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 2: Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        tex_filename = f"paper_{timestamp}.tex"

        # Step 3: Save LaTeX content
        tex_file = output_dir / tex_filename

        tex_file.write_text(latex_content,encoding="utf-8")

        # Step 4: Verify file creation
        if not tex_file.is_file():
            raise FileNotFoundError(f"LaTeX file was not generated: {tex_file}")

        print(f"Successfully saved LaTeX file at: {tex_file}")

        # Step 5: Return file path
        return str(tex_file)

    except Exception as e:
        print(f"Error saving LaTeX: {str(e)}")
        raise


# latex_content = r"""
# \documentclass{article}
# \usepackage{amsmath}
# \begin{document}
# \title{Machine Learning Research Paper}
# \author{AI Research Agent}
# \maketitle

# \section{Introduction}
# Machine learning is a branch of artificial intelligence.
# \section{Mathematical Formulation}
# \begin{equation}
# E = mc^2
# \end{equation}
# \section{Conclusion}
# Machine learning enables intelligent predictions.
# \end{document}
# """

# result = render_latex_code.invoke({"latex_content": latex_content})
# print(result)