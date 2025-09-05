import re
import xml.etree.ElementTree as ET

def clean_wiki_text(text: str) -> str:
    """Remove Wikipedia markup and return clean text."""
    if not text:
        return ""

    # Remove templates {{...}}
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)

    # Remove file/image links [[File:...]] or [[Image:...]]
    text = re.sub(r"\[\[(File|Image):.*?\]\]", "", text, flags=re.IGNORECASE)

    # Remove category links [[Category:...]]
    text = re.sub(r"\[\[Category:.*?\]\]", "", text, flags=re.IGNORECASE)

    # Remove external links [http://... label]
    text = re.sub(r"\[https?:\/\/[^\s\]]+(\s+[^\]]+)?\]", "", text)

    # Remove internal links [[...|...]] → keep only the label
    text = re.sub(r"\[\[([^\|\]]*\|)?([^\]]+)\]\]", r"\2", text)

    # Remove any remaining [[...]]
    text = re.sub(r"\[\[(.*?)\]\]", r"\1", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove multiple newlines
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


def extract_wiki_dump(xml_file: str, output_file: str):
    """Extract plain text from a Wikipedia XML dump (.xml)."""
    tree = ET.iterparse(xml_file, events=("end",))
    with open(output_file, "w", encoding="utf-8") as out:
        for event, elem in tree:
            if elem.tag.endswith("text"):
                raw_text = elem.text
                clean_text = clean_wiki_text(raw_text)
                if clean_text:
                    out.write(clean_text + "\n\n")
                elem.clear()  # free memory


if __name__ == "__main__":
    input_xml = "simplewiki-latest-pages-articles-multistream.xml"
    output_txt = "simplewiki_clean.txt"
    print(f"Extracting {input_xml} → {output_txt} ...")
    extract_wiki_dump(input_xml, output_txt)
    print("Done ✅")