def generate_pdf_html(project_name, badge_html, sections):
    """
    Create a full HTML string ready for PDF generation
    badge_html: HTML for badges (MIT, Python, Date, AI mode)
    sections: list of tuples [(title, html_content)]
    """
    toc = "<h2>Table of Contents</h2><ul>"
    for i, (title, _) in enumerate(sections, start=1):
        toc += f'<li><a href="#section{i}">{title}</a></li>'
    toc += "</ul><hr>"

    body = ""
    for i, (title, content) in enumerate(sections, start=1):
        body += f'<h2 id="section{i}">{title}</h2>\n{content}\n<hr>'

    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ text-align: center; }}
            h2 {{ color: #2F4F4F; margin-top: 40px; }}
            hr {{ border: 0; border-top: 1px solid #ccc; margin: 20px 0; }}
            ul {{ list-style-type: none; }}
            li {{ margin-bottom: 5px; }}
            footer {{ position: fixed; bottom: 0; width: 100%; text-align: center; font-size: 10px; color: gray; }}
        </style>
    </head>
    <body>
        <h1>{project_name}</h1>
        <div>{badge_html}</div>
        {toc}
        {body}
        <footer>Page <span class="pageNumber"></span></footer>
    </body>
    </html>
    """
    return html
