from datetime import datetime

def convert_yyyy_mmm_to_yyyy_mm(date_str):
    month_map = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    year, month_abbr = date_str.split('-')
    month = month_map[month_abbr]
    return f"{year}-{month}"