"""
generate_dataset.py
--------------------
Builds data/news_dataset.csv used to train the fake-news classifier.

Real news templates mimic wire-service style: attributed sources, hedged
language, dates/figures, official titles.
Fake news templates mimic clickbait/misinformation style: absolute claims,
emotional triggers, unnamed sources, ALL-CAPS emphasis, conspiracy phrasing.
"""

import csv
import os
import random

random.seed(42)

TOPICS = [
    "the economy", "a new vaccine", "the education system", "climate policy",
    "a local election", "the housing market", "a tech company merger",
    "the national football league", "a celebrity marriage", "space exploration",
    "the stock market", "a public health study", "immigration policy",
    "a government budget bill", "artificial intelligence regulation",
    "a natural disaster", "the police department", "a university research team",
    "the central bank", "a food safety recall", "renewable energy",
    "a cybersecurity breach", "the healthcare system", "a court ruling",
    "the transportation department", "a wildlife conservation project",
]

REAL_TEMPLATES = [
    "Officials confirmed on Tuesday that {topic} will undergo review after a report was submitted to the committee.",
    "According to a statement released by the department, {topic} showed a 3.2 percent change compared to last quarter.",
    "Researchers at the state university published findings on {topic} in a peer-reviewed journal this week.",
    "The city council voted 6-3 on Monday to approve new measures related to {topic}, pending final approval.",
    "A spokesperson said the agency is continuing to monitor {topic} and will release further details next month.",
    "Data released by the bureau indicates that {topic} remained largely stable over the past year, analysts said.",
    "The committee held a public hearing on {topic}, inviting testimony from experts and community members.",
    "Local authorities announced a new initiative concerning {topic}, which will be implemented over the next six months.",
    "In a press briefing on Thursday, the minister addressed questions about {topic} and outlined next steps.",
    "The report, compiled over eighteen months, found no significant risk associated with {topic} based on current evidence.",
    "Company executives confirmed the deal involving {topic} during an earnings call with investors on Wednesday.",
    "The study, funded by a federal grant, examined {topic} using data collected from over 2,000 participants.",
    "A federal judge issued a ruling on {topic} on Friday, siding with the plaintiffs on two of three counts.",
    "The organization released its annual report on {topic}, noting modest improvements year over year.",
    "Emergency services responded to the incident related to {topic} and confirmed there were no injuries reported.",
    "Economists at the central bank projected {topic} would grow modestly in the coming fiscal year.",
    "The mayor's office issued a statement clarifying its position on {topic} following community feedback sessions.",
    "A panel of independent auditors reviewed {topic} and recommended several procedural changes.",
    "Health officials advised residents to stay informed about {topic} through verified government channels.",
    "The university's board of trustees discussed {topic} during its quarterly meeting on Monday afternoon.",
]

FAKE_TEMPLATES = [
    "SHOCKING: You won't believe what they're hiding about {topic} — doctors HATE this one secret!",
    "BREAKING: Anonymous insider reveals {topic} is a total SCAM designed to control the population!!!",
    "They don't want you to know the TRUTH about {topic}. Share before this gets DELETED!",
    "Experts are STUNNED after leaked documents expose the real agenda behind {topic}.",
    "This ONE weird trick about {topic} has the entire government in a panic, sources claim.",
    "URGENT: {topic} has secretly been rigged for years, according to a whistleblower nobody wants you to hear.",
    "Wake up! {topic} is nothing but a cover-up orchestrated by powerful elites behind closed doors.",
    "You will NEVER guess what really happened with {topic} — mainstream media REFUSES to report this!",
    "A mysterious source close to the matter says {topic} was staged from the very beginning.",
    "Doctors are BAFFLED: the hidden dangers of {topic} that 'they' don't want exposed.",
    "EXCLUSIVE: Secret files prove {topic} was planned decades in advance by a shadowy group.",
    "Is {topic} actually a hoax? This viral post claims 100% proof and everyone is sharing it.",
    "The REAL reason behind {topic} will make your blood boil — click to see the shocking evidence!",
    "Insiders CONFIRM {topic} is being used to secretly manipulate everyday citizens like you.",
    "This changes EVERYTHING: unverified report claims {topic} was faked using actors.",
    "Government sources who wish to remain anonymous say {topic} is far worse than reported.",
    "People are FURIOUS after discovering the hidden truth about {topic} that was covered up for years.",
    "You've been LIED to about {topic} this whole time — share this before it's taken down!",
    "Miracle claim goes viral: {topic} allegedly linked to a secret plot nobody can explain.",
    "REVEALED: The disturbing conspiracy behind {topic} that has the internet in total shock.",
]


def build_rows():
    rows = []
    for topic in TOPICS:
        for tmpl in REAL_TEMPLATES:
            rows.append((tmpl.format(topic=topic), "REAL"))
        for tmpl in FAKE_TEMPLATES:
            rows.append((tmpl.format(topic=topic), "FAKE"))
    random.shuffle(rows)
    return rows


def main():
    rows = build_rows()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "news_dataset.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    real_count = sum(1 for _, l in rows if l == "REAL")
    fake_count = sum(1 for _, l in rows if l == "FAKE")
    print(f"REAL: {real_count}  FAKE: {fake_count}")


if __name__ == "__main__":
    main()
