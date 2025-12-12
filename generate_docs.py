import os
from datetime import datetime

from modules.kickoff import generate_kickoff_summary
from modules.notes import generate_notes_summary
from modules.summary import generate_final_summary
from modules.tasks import extract_tasks
from modules.risks import extract_risks
from shared.md_header import generate_md_header
from shared.pdf_utils import generate_pdf_html


import markdown      # <-- required for markdown.markdown()
import yaml          # <-- required for YAML metadata loading
from weasyprint import HTML


# -----------------------
# INPUT SECTION
# -----------------------

project_name = input("Enter project name: ").strip()
objectives = input("Enter project objectives: ").strip()
raw_notes = input("Enter project notes: ").strip()
accomplishments = input("Enter accomplishments: ").strip()
lessons_learned = input("Enter lessons learned: ").strip()

date = datetime.now().strftime("%Y-%m-%d")


# -----------------------
# OUTPUT FOLDER
# -----------------------

output_folder = "examples"
os.makedirs(output_folder, exist_ok=True)


# -----------------------
# GENERATE KICKOFF SUMMARY
# -----------------------

kickoff_content, kickoff_mode = generate_kickoff_summary(project_name, objectives)
kickoff_header = generate_md_header(project_name, objectives, date, kickoff_mode)

with open(os.path.join(output_folder, "kickoff_summary.md"), "w") as f:
    f.write(kickoff_header + kickoff_content)


# -----------------------
# GENERATE NOTES SUMMARY
# -----------------------

notes_content, notes_mode = generate_notes_summary(raw_notes, date)
notes_header = generate_md_header(project_name, objectives, date, notes_mode)

with open(os.path.join(output_folder, "daily_notes_summary.md"), "w") as f:
    f.write(notes_header + notes_content)


# -----------------------
# GENERATE FINAL SUMMARY
# -----------------------

final_content, final_mode = generate_final_summary(project_name, accomplishments, lessons_learned)
final_header = generate_md_header(project_name, objectives, date, final_mode)

with open(os.path.join(output_folder, "final_project_summary.md"), "w") as f:
    f.write(final_header + final_content)


# -----------------------
# GENERATE TASKS
# -----------------------

tasks_content, tasks_mode = extract_tasks(raw_notes)
tasks_header = generate_md_header(project_name, objectives, date, tasks_mode)

with open(os.path.join(output_folder, "extracted_tasks.md"), "w") as f:
    f.write(tasks_header + tasks_content)


# -----------------------
# GENERATE RISKS
# -----------------------

risks_content, risks_mode = extract_risks(raw_notes)
risks_header = generate_md_header(project_name, objectives, date, risks_mode)

with open(os.path.join(output_folder, "extracted_risks.md"), "w") as f:
    f.write(risks_header + risks_content)


print("✅ All documents generated with dynamic metadata & badges!")
print(f"📁 Saved in: {output_folder}/")
# -----------------------
# HTML / PDF REPORT GENERATION
# -----------------------

sections = []

badge_html = (
    '<img src="https://img.shields.io/badge/License-MIT-green"> '
    '<img src="https://img.shields.io/badge/Python-3.11-blue"> '
    f'<img src="https://img.shields.io/badge/Last%20Update-{date}-yellow"> '
    '<img src="https://img.shields.io/badge/Content-Placeholder-orange">'
)

md_files = [
    "kickoff_summary.md",
    "daily_notes_summary.md",
    "final_project_summary.md",
    "extracted_tasks.md",
    "extracted_risks.md"
]

for md_file in md_files:
    md_path = os.path.join(output_folder, md_file)
    with open(md_path, "r") as f:
        md_text = f.read()
        html_content = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
        title = md_file.replace('_', ' ').replace('.md', '').title()
        sections.append((title, html_content))

# Generate full HTML for PDF
full_html = generate_pdf_html(project_name, badge_html, sections)

# Save the professional PDF
pdf_path = os.path.join(output_folder, f"{project_name.replace(' ', '_')}_professional_report.pdf")
HTML(string=full_html).write_pdf(pdf_path)

print(f"✅ Professional PDF report saved: {pdf_path}")
