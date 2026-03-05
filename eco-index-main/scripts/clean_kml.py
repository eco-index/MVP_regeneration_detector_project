import argparse
import os
import xml.etree.ElementTree as ET

STYLE_ID = "data-collection"
LINE_COLOR = "ffff00ff"
POLY_COLOR = "ffff00ff"
LINE_WIDTH = "0.5"

NS = {"kml": "http://www.opengis.net/kml/2.2"}
ET.register_namespace("", NS["kml"])

def clean_kml(input_path: str, output_path: str) -> None:
    tree = ET.parse(input_path)
    root = tree.getroot()
    doc = root.find("kml:Document", NS)
    if doc is None:
        doc = root

    # Remove existing style definitions
    for tag in ["StyleMap", "Style"]:
        for elem in doc.findall(f"kml:{tag}", NS):
            doc.remove(elem)

    # Insert consistent style definition
    style = ET.Element("Style", id=STYLE_ID)
    line_style = ET.SubElement(style, "LineStyle")
    width = ET.SubElement(line_style, "width")
    width.text = LINE_WIDTH
    line_color = ET.SubElement(line_style, "color")
    line_color.text = LINE_COLOR
    poly_style = ET.SubElement(style, "PolyStyle")
    poly_color = ET.SubElement(poly_style, "color")
    poly_color.text = POLY_COLOR

    # insert style after the <name> and <Schema> elements if present
    insert_index = 0
    for i, child in enumerate(list(doc)):
        tag = child.tag.split('}')[-1]
        if tag in {"name", "Schema"}:
            insert_index = i + 1
    doc.insert(insert_index, style)

    # ensure folders are open and free of <visibility> tags
    for folder in doc.findall('.//kml:Folder', NS):
        for vis in folder.findall('kml:visibility', NS):
            folder.remove(vis)
        open_tag = folder.find('kml:open', NS)
        if open_tag is None:
            open_tag = ET.Element('open')
            open_tag.text = '1'
            # insert after <name> if present
            insert_idx = 0
            for i, child in enumerate(list(folder)):
                if child.tag.split('}')[-1] == 'name':
                    insert_idx = i + 1
                    break
            folder.insert(insert_idx, open_tag)
        else:
            open_tag.text = '1'

    # Update each Placemark
    for placemark in doc.findall(".//kml:Placemark", NS):
        for vis in placemark.findall("kml:visibility", NS):
            placemark.remove(vis)
        for inline_style in placemark.findall("kml:Style", NS):
            placemark.remove(inline_style)
        style_url = placemark.find("kml:styleUrl", NS)
        if style_url is None:
            style_url = ET.Element("styleUrl")
            placemark.insert(0, style_url)
        style_url.text = f"#{STYLE_ID}"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    xml_bytes = ET.tostring(root, encoding="utf-8")
    try:
        from xml.dom import minidom

        parsed = minidom.parseString(xml_bytes)
        pretty = parsed.toprettyxml(indent="    ", encoding="utf-8")
        with open(output_path, "wb") as f:
            f.write(pretty)
    except Exception:
        tree.write(output_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean KML styling and visibility")
    parser.add_argument("--input", required=True, help="Path to raw KML file")
    parser.add_argument("--output", required=True, help="Path to write cleaned KML")
    args = parser.parse_args()
    clean_kml(args.input, args.output)


if __name__ == "__main__":
    main()
