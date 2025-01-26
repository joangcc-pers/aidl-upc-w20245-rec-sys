from enum import Enum

class CsvFilesEnum(Enum):
    # Define keys and values for months from October 2019 to June 2020
    @staticmethod
    def from_date(date_str: str) -> str:
        month_map = {
            "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
            "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
            "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"
        }
        if not isinstance(date_str, str) or len(date_str) != 7 or date_str[4] != "-" or not date_str[:4].isdigit() or not date_str[5:].isdigit():
            raise ValueError("Invalid date format. Expected format is 'YYYY-MM'.")
        year, month = date_str.split("-")
        if month not in month_map:
            raise ValueError(f"Invalid month '{month}' in input. Expected values are between '01' and '12'.")
        return f"{year}-{month_map[month]}.csv"

    OCT_2019 = "2019-Oct.csv"
    NOV_2019 = "2019-Nov.csv"
    DEC_2019 = "2019-Dec.csv"
    JAN_2020 = "2020-Jan.csv"
    FEB_2020 = "2020-Feb.csv"
    MAR_2020 = "2020-Mar.csv"
    APR_2020 = "2020-Apr.csv"
    MAY_2020 = "2020-May.csv"
    JUN_2020 = "2020-Jun.csv"