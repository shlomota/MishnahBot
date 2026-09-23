"""Daily Mishnah calendar: maps Hebrew dates to today's chapters and Sefaria refs.

Calendar data is a static snapshot of R. Ethan Tucker's Hebrew-year-aligned
Mishnah learning schedule (Hebrew year 5786), stored in mishnah_calendar.csv.
Days are matched by Hebrew month/day (ignoring year) so the same 354-day cycle
can be looked up regardless of which Gregorian/Hebrew year it's currently rendered in.
"""
import csv
import os
import re
from dataclasses import dataclass, field

from pyluach import dates

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mishnah_calendar.csv")

# pyluach's month names -> this calendar's month names
MONTH_NAME_MAP = {
    "Nissan": "Nisan",
    "Teves": "Tevet",
    "Cheshvan": "Marheshvan",
    "Adar 1": "Adar",
    "Adar 2": "Adar",
    "Adar": "Adar",
}

# CSV tractate name -> canonical Sefaria title (spellings differ between the
# calendar source and Sefaria's indexing).
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


def _load_calendar():
    days = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            day_num = row.get("Day #", "").strip()
            if not day_num.isdigit():
                continue
            readings, is_siyum = _parse_schedule(row.get("Schedule", ""))
            days.append(
                CalendarDay(
                    day_num=int(day_num),
                    hebrew_date=row.get("Hebrew Date", "").strip(),
                    weekday=row.get("Weekday", "").strip(),
                    raw_schedule=(row.get("Schedule") or "").strip(),
                    parsha=(row.get("Parsha (Diaspora 5786)") or "").strip(),
                    completes=(row.get("Massekhot to Be Completed by this date") or "").strip(),
                    readings=readings,
                    is_siyum=is_siyum,
                )
            )
    return days


CALENDAR_DAYS = _load_calendar()
TOTAL_DAYS = len(CALENDAR_DAYS)
_BY_HEBREW_DATE = {d.hebrew_date: d for d in CALENDAR_DAYS}
_BY_DAY_NUM = {d.day_num: d for d in CALENDAR_DAYS}


def hebrew_date_str_for(gregorian_date):
    """Convert a datetime.date to this calendar's 'D Month' string, e.g. '11 Tishrei'."""
    heb = dates.GregorianDate(gregorian_date.year, gregorian_date.month, gregorian_date.day).to_heb()
    month_name = heb.month_name(hebrew=False)
    month_name = MONTH_NAME_MAP.get(month_name, month_name)
    return f"{heb.day} {month_name}"


def day_for_date(gregorian_date):
    """Return the CalendarDay matching a Gregorian date's Hebrew month/day, if any."""
    return _BY_HEBREW_DATE.get(hebrew_date_str_for(gregorian_date))


def get_day(day_num):
    return _BY_DAY_NUM.get(day_num)
