from urllib import request
from bs4 import BeautifulSoup
from unidecode import unidecode
import textwrap
from fpdf import FPDF

def text_to_pdf(text, filename):
    a4_width_mm = 210
    pt_to_mm = 0.35
    fontsize_pt = 10
    fontsize_mm = fontsize_pt * pt_to_mm
    margin_bottom_mm = 10
    character_width_mm = 7 * pt_to_mm
    width_text = a4_width_mm / character_width_mm

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(True, margin=margin_bottom_mm)
    pdf.add_page()
    pdf.set_font(family='Courier', size=fontsize_pt)
    splitted = text.split('\n')

    for line in splitted:
        lines = textwrap.wrap(line, width_text)

        if len(lines) == 0:
            pdf.ln()

        for wrap in lines:
            pdf.cell(0, fontsize_mm, wrap, ln=1)

    pdf.output(filename, 'F')

def extract_html_to_pdf(url):
    html = request.urlopen(url).read().decode('utf8')

    soup = BeautifulSoup(html, 'html.parser')
    title = soup.find('title')
    body = soup.find('div', {'class': 'blog-post__body-wrapper'})

    with open("temp.txt", "w") as text_file:
        text_file.write(unidecode(title.text) + unidecode(body.text))

    title = title.text.replace(':','')
        
    output_filename = f'Docs/Webpage_{title}.pdf'
    file = open('temp.txt')
    text = file.read()
    file.close()
    text_to_pdf(text, output_filename)

def main():
    with open('URLs.txt', 'r') as f:
        contents = f.readlines()

    for url in contents:
        extract_html_to_pdf(url)

if __name__=='__main__':
    main()
