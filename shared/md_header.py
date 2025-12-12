from datetime import datetime


def generate_md_header(project_name, objectives, date, ai_mode="Placeholder"):
    """
    Returns a formatted Markdown header with badges and project metadata.
    
    ai_mode options:
    - "OpenAI"
    - "Placeholder"
    """

    # Format date safely for badge encoding (no spaces)
    date_for_badge = date.replace(" ", "%20")

    return f"""# {project_name}

![License](https://img.shields.io/badge/License-MIT-green)
![Python Version](https://img.shields.io/badge/Python-3.11-blue)
![Last Update](https://img.shields.io/badge/Last%20Update-{date_for_badge}-yellow)
![Content Mode](https://img.shields.io/badge/Content-{ai_mode}-orange)

**Project Objectives:**  
{objectives}

---
"""
