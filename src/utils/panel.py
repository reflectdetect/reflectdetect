def get_panel_factors_for_band(panel_data, band):
    return [panel["bands"][band]["factor"] for panel in panel_data]