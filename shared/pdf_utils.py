def generate_pdf_html(project_name, badges_html, sections):
    """
    Build a professional HTML project report with TOC and styled sections.
    """

    # Build a table of contents
    toc_html = "<ul>"
    for title, _ in sections:
        anchor = title.lower().replace(" ", "-")
        toc_html += f'<li><a href="#{anchor}">{title}</a></li>'
    toc_html += "</ul>"

    # Build the sections
    sections_html = ""
    for title, content in sections:
        anchor = title.lower().replace(" ", "-")
        sections_html += f"""
        <div class="section" id="{anchor}">
            <h2>{title}</h2>
            <div class="content-block">
                {content}
            </div>
        </div>
        """

    # Return full HTML document
    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{project_name} – Executive Project Report</title>
        <style>

            body {{
                font-family: Arial, sans-serif;
                padding: 40px;
                line-height: 1.6;
                background: #fafafa;
                color: #222;
                max-width: 900px;
                margin: auto;
            }}

            h1 {{
                font-size: 36px;
                margin-bottom: 10px;
            }}

            h2 {{
                font-size: 26px;
                border-bottom: 2px solid #ddd;
                padding-bottom: 4px;
                margin-top: 40px;
            }}

            .badges {{
                margin-top: 10px;
                margin-bottom: 30px;
            }}

            .toc {{
                background: #fff;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #e5e5e5;
                box-shadow: 0px 4px 8px rgba(0,0,0,0.04);
                margin-bottom: 40px;
            }}

            .toc h3 {{
                margin-top: 0;
            }}

            .section {{
                background: #fff;
                padding: 25px;
                border-radius: 10px;
                margin-bottom: 30px;
                border: 1px solid #e1e1e1;
                box-shadow: 0px 4px 8px rgba(0,0,0,0.03);
            }}

            .content-block {{
                margin-top: 10px;
            }}

            a {{
                color: #0066cc;
                text-decoration: none;
            }}

            a:hover {{
                text-decoration: underline;
            }}

        </style>
    </head>
    <body>

        <h1>{project_name}</h1>
        <div class="badges">{badges_html}</div>

        <div class="toc">
            <h3>Table of Contents</h3>
            {toc_html}
        </div>

        {sections_html}

    </body>
    </html>
    """

    return html
