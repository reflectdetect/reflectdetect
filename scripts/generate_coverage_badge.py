import xml.etree.ElementTree


def get_coverage_percentage():
    tree = xml.etree.ElementTree.parse('coverage.xml')
    root = tree.getroot()
    cov = root.attrib['line-rate']
    return float(cov) * 100


if __name__ == "__main__":
    coverage = get_coverage_percentage()
    with open('badge_url.txt', 'w', encoding='utf-8') as f:
        f.write(str(int(coverage)))
