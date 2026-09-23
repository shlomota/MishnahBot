"""Daily Mishnah calendar: maps Hebrew dates to today's chapters and Sefaria refs.

Calendar data is a static snapshot of R. Ethan Tucker's Hebrew-year-aligned
Mishnah learning schedule. A Hebrew year's shape is fully determined by two
facts: the weekday its 1 Tishrei falls on, and its length in days (353-355 for
a regular year, 383-385 for a leap year). There are exactly 14 valid
combinations ("keviah" patterns), and the source spreadsheet has one tab per
pattern (named things like "7F5"), each a complete day-by-day schedule for any
Hebrew year matching that pattern. We pick the matching tab for whichever
Hebrew year is relevant and look up within it by Hebrew month/day - so the
calendar stays correct across Hebrew years without any manual updates.
"""
import csv
import glob
import os
import re
from dataclasses import dataclass, field

from pyluach import dates

VARIANTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mishnah_calendar_variants")

# pyluach's month names -> this calendar's month names
MONTH_NAME_MAP = {
    "Nissan": "Nisan",
    "Teves": "Tevet",
    "Cheshvan": "Marheshvan",
    "Adar 1": "Adar I",
    "Adar 2": "Adar II",
    "Adar": "Adar",
}

# CSV tractate name -> canonical Sefaria title (spellings differ between the
# calendar source and Sefaria's indexing, and even between variant tabs).
TRACTATE_TO_SEFARIA = {
    "Aholot": "Mishnah Oholot",
    "Arakhin": "Mishnah Arakhin",
    "Avodah Zarah": "Mishnah Avodah Zarah",
    "Avot": "Pirkei Avot",
    "Bava Batra": "Mishnah Bava Batra",
    "Bava Kama": "Mishnah Bava Kamma",
    "Bava Metzia": "Mishnah Bava Metzia",
    "Beitzah": "Mishnah Beitzah",
    "Bekhorot": "Mishnah Bekhorot",
    "Berakhot": "Mishnah Berakhot",
    "Bikkurim": "Mishnah Bikkurim",
    "Demai": "Mishnah Demai",
    "Eduyot": "Mishnah Eduyot",
    "Eruvin": "Mishnah Eruvin",
    "Gittin": "Mishnah Gittin",
    "Hagigah": "Mishnah Chagigah",
    "Hallah": "Mishnah Challah",
    "Horayot": "Mishnah Horayot",
    "Hullin": "Mishnah Chullin",
    "Keilim": "Mishnah Kelim",
    "Kereitot": "Mishnah Keritot",
    "Ketubot": "Mishnah Ketubot",
    "Kiddushin": "Mishnah Kiddushin",
    "Kilayim": "Mishnah Kilayim",
    "Kinnim": "Mishnah Kinnim",
    "Ma'aser Sheni": "Mishnah Maaser Sheni",
    "Ma'aserot": "Mishnah Maasrot",
    "Makkot": "Mishnah Makkot",
    "Makshirin": "Mishnah Makhshirin",
    "Me'ilah": "Mishnah Meilah",
    "Meilah": "Mishnah Meilah",
    "Megillah": "Mishnah Megillah",
    "Menahot": "Mishnah Menachot",
    "Middot": "Mishnah Middot",
    "Mikvaot": "Mishnah Mikvaot",
    "Moed Katan": "Mishnah Moed Katan",
    "Nazir": "Mishnah Nazir",
    "Nedarim": "Mishnah Nedarim",
    "Negaim": "Mishnah Negaim",
    "Niddah": "Mishnah Niddah",
    "Orlah": "Mishnah Orlah",
    "Parah": "Mishnah Parah",
    "Peah": "Mishnah Peah",
    "Pesahim": "Mishnah Pesachim",
    "Rosh Hashanah": "Mishnah Rosh Hashanah",
    "Sanhedrin": "Mishnah Sanhedrin",
    "Shabbat": "Mishnah Shabbat",
    "Shekalim": "Mishnah Shekalim",
    "Sheviit": "Mishnah Sheviit",
    "Shevuot": "Mishnah Shevuot",
    "Sotah": "Mishnah Sotah",
    "Sukkah": "Mishnah Sukkah",
    "Taanit": "Mishnah Ta'anit",
    "Tahorot": "Mishnah Tahorot",
    "Tamid": "Mishnah Tamid",
    "Temurah": "Mishnah Temurah",
    "Terumot": "Mishnah Terumot",
    "Tevul Yom": "Mishnah Tevul Yom",
    "Uktzin": "Mishnah Oktzin",
    "Yadayim": "Mishnah Yadayim",
    "Yevamot": "Mishnah Yevamot",
    "Yoma": "Mishnah Yoma",
    "Zavim": "Mishnah Zavim",
    "Zevahim": "Mishnah Zevachim",
}

CHAPTER_RE = re.compile(r"^(.+?)\s+(\d+)(?:-(\d+))?$")


@dataclass
class ChapterReading:
    tractate: str
    sefaria_title: str
    chapter_start: int
    chapter_end: int

    @property
    def label(self):
        if self.chapter_end != self.chapter_start:
            return f"{self.tractate} {self.chapter_start}-{self.chapter_end}"
        return f"{self.tractate} {self.chapter_start}"

    @property
    def sefaria_ref(self):
        slug = self.sefaria_title.replace(" ", "_")
        if self.chapter_end != self.chapter_start:
            return f"{slug}.{self.chapter_start}-{self.chapter_end}"
        return f"{slug}.{self.chapter_start}"

    @property
    def sefaria_url(self):
        return f"https://www.sefaria.org/{self.sefaria_ref}?lang=bi&with=Commentary%20ConnectionsList&lang2=en"


@dataclass
class CalendarDay:
    day_num: int
    hebrew_date: str
    weekday: str
    raw_schedule: str
    parsha: str
    completes: str
    readings: list = field(default_factory=list)
    is_siyum: bool = False


def _normalize_tractate_name(name):
    return " ".join(name.strip().split())


def _parse_schedule(raw_schedule):
    """Turn a raw Schedule cell into a list of ChapterReading (or mark siyum)."""
    raw_schedule = (raw_schedule or "").strip()
    if not raw_schedule:
        return [], False
    if "SIYYUM" in raw_schedule.upper():
        return [], True

    readings = []
    for part in re.split(r"[;,]", raw_schedule):
        part = part.strip()
        if not part:
            continue
        m = CHAPTER_RE.match(part)
        if not m:
            continue
        tractate = _normalize_tractate_name(m.group(1))
        start = int(m.group(2))
        end = int(m.group(3)) if m.group(3) else start
        sefaria_title = TRACTATE_TO_SEFARIA.get(tractate, f"Mishnah {tractate}")
        readings.append(ChapterReading(tractate, sefaria_title, start, end))
    return readings, False


def _load_variant(csv_path):
    days = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            day_num = (row.get("Day #") or "").strip()
            if not day_num.isdigit():
                continue
            readings, is_siyum = _parse_schedule(row.get("Schedule", ""))
            parsha = (row.get("Parsha") or "").replace("\n", " / ").strip()
            days.append(
                CalendarDay(
                    day_num=int(day_num),
                    hebrew_date=(row.get("Hebrew Date") or "").strip(),
                    weekday=(row.get("Weekday") or "").strip(),
                    raw_schedule=(row.get("Schedule") or "").strip(),
                    parsha=parsha,
                    completes=(row.get("Completes") or "").strip(),
                    readings=readings,
                    is_siyum=is_siyum,
                )
            )
    return days


def _load_all_variants():
    variants = {}
    for path in sorted(glob.glob(os.path.join(VARIANTS_DIR, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        days = _load_variant(path)
        if not days:
            continue
        signature = (days[0].weekday, len(days))
        variants[name] = {"days": days, "signature": signature}
    return variants


_VARIANTS = _load_all_variants()
_SIGNATURE_TO_VARIANT = {v["signature"]: name for name, v in _VARIANTS.items()}


def _hebrew_year_signature(hebrew_year):
    """The (weekday-of-1-Tishrei, year-length-in-days) signature for a Hebrew year."""
    start = dates.HebrewDate(hebrew_year, 7, 1).to_pydate()  # 7 = Tishrei
    next_start = dates.HebrewDate(hebrew_year + 1, 7, 1).to_pydate()
    weekday = start.strftime("%A")
    length = (next_start - start).days
    return weekday, length


def variant_for_hebrew_year(hebrew_year):
    """Return (variant_name, list[CalendarDay]) for the given Hebrew year."""
    signature = _hebrew_year_signature(hebrew_year)
    variant_name = _SIGNATURE_TO_VARIANT.get(signature)
    if variant_name is None:
        raise ValueError(f"No calendar variant found for Hebrew year {hebrew_year} (signature {signature})")
    return variant_name, _VARIANTS[variant_name]["days"]


def hebrew_date_str_for(gregorian_date):
    """Convert a datetime.date to this calendar's 'D Month' string, e.g. '11 Tishrei'."""
    heb = dates.GregorianDate(gregorian_date.year, gregorian_date.month, gregorian_date.day).to_heb()
    month_name = heb.month_name(hebrew=False)
    month_name = MONTH_NAME_MAP.get(month_name, month_name)
    return heb.year, f"{heb.day} {month_name}"


def calendar_for_date(gregorian_date):
    """Return (cycle_id, variant_name, days, day_for_this_date_or_None) for a Gregorian date."""
    hebrew_year, hebrew_date_str = hebrew_date_str_for(gregorian_date)
    variant_name, days = variant_for_hebrew_year(hebrew_year)
    by_hebrew_date = {d.hebrew_date: d for d in days}
    return str(hebrew_year), variant_name, days, by_hebrew_date.get(hebrew_date_str)


def _month_name_to_num(hebrew_year):
    """This calendar's month names -> pyluach month numbers for a given Hebrew year.

    Built by probing every valid month number for the year rather than
    hardcoding a numbering scheme, since that shifts between leap and
    non-leap years (an extra Adar).
    """
    mapping = {}
    for month_num in range(1, 14):
        try:
            hd = dates.HebrewDate(hebrew_year, month_num, 1)
        except ValueError:
            continue
        raw_name = hd.month_name(hebrew=False)
        mapping[MONTH_NAME_MAP.get(raw_name, raw_name)] = month_num
    return mapping


def gregorian_date_for(hebrew_year, hebrew_date_str):
    """Convert this calendar's 'D Month' string (for a given Hebrew year) to a datetime.date."""
    day_str, month_name = hebrew_date_str.split(" ", 1)
    month_num = _month_name_to_num(hebrew_year).get(month_name)
    if month_num is None:
        return None
    return dates.HebrewDate(hebrew_year, month_num, int(day_str)).to_pydate()
