import xml.etree.ElementTree as ET


def get_coverage_percentage():
    tree = ET.parse('coverage.xml')
    root = tree.getroot()
    coverage = root.attrib['line-rate']
    return float(coverage) * 100


def generate_badge(coverage):
    badge_url = f"https://img.shields.io/badge/Coverage-{coverage}%25-brightgreen"
    return badge_url


if __name__ == "__main__":
    coverage = get_coverage_percentage()
    badge_url = generate_badge(int(coverage))
    print(f"Coverage Badge URL: {badge_url}")
