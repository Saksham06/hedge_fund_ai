from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def generate_pdf(text, filename="report.pdf"):

    doc = SimpleDocTemplate(filename)
    story = []
    style = getSampleStyleSheet()["BodyText"]

    for line in text.split("\n"):
        safe = (line or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe, style))
        story.append(Spacer(1, 10))

    doc.build(story)
