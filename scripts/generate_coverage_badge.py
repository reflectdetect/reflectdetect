import xml.etree.ElementTree as ET


def get_coverage_percentage():
    tree = ET.parse('coverage.xml')
    root = tree.getroot()
    coverage = root.attrib['line-rate']
    return float(coverage) * 100


if __name__ == "__main__":
    coverage = get_coverage_percentage()
    with open('badge_url.txt', 'w', encoding='utf-8') as f:
        f.write(str(int(coverage)))
