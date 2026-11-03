"""Generate data/soft_skill_labeled.csv (reproducible, seeded).

HOW THIS DATASET WAS MADE (also stated in data/DATASET_CARD.md):

* The question list and, for each question, hand-authored answer *fragments*
  (situation / action / result / reflection sentences, plus average and poor answers)
  were written by an LLM (Claude) during project scaffolding.
* This script composes fragments into full answers with a fixed random seed:
    good    = situation + action + result (+ reflection)   -> specific, structured, measurable
    average = hand-written generic answers, or situation + action only (no result)
    poor    = short, vague, hedged or filler-heavy answers
* 24 extra "length-matched distractors" (12 average + 12 poor) pad a weak answer with generic or
  off-topic sentences until it is as long as a good answer. Without them, answer length alone
  separates the classes (a word-count-only model scored ~94%), so a classifier would learn nothing
  about content.
* Every row has ``reviewed = no``. A human reviewer is expected to read each row, fix the
  label if needed and set ``reviewed = yes`` (see the "Review workflow" in the dataset card).
  Until that happens the dataset is *LLM-authored and unreviewed*.

To grow the dataset, add fragments below (or generate more with your LLM provider and
append reviewed rows to the CSV) and re-run:  python scripts/generate_soft_skill_dataset.py
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

SEED = 42
PER_LABEL = 5  # rows per question per label -> 12 questions x 3 labels x 5 = 180 rows
PAD_AVG = [
    "I believe that communication and teamwork are very important in every project and I always try to improve them.",
    "In general I try to stay positive, work hard and keep learning from the people around me.",
    "I think every experience helps a person grow, and I always try to take the responsibility given to me seriously.",
    "Overall it was a good experience for me and it helped me understand how things work in a real environment.",
]
PAD_POOR = [
    "Honestly in life everything happens for a reason, and I try to stay positive about whatever comes.",
    "Many people say that hard work is the key to success and I also agree with this opinion completely.",
    "I believe that attitude matters more than anything else, and that is how I look at most things.",
    "Technology is changing very fast these days, so everybody needs to keep adapting to the new world.",
]
OUT = Path(__file__).resolve().parents[1] / "data" / "soft_skill_labeled.csv"

# Each entry: category, question, S (situations), A (actions), R (results), L (reflections),
# AVG (average answers), POOR (poor answers)
Q = [
    dict(
        id="b1", category="behavioral",
        question="Tell me about a time you had a disagreement with a teammate. How did you resolve it?",
        S=["In our third-year database project, my teammate wanted MongoDB while I preferred MySQL, and the argument stalled our design for a week.",
           "During a hackathon, a teammate and I disagreed on whether to build the mobile app first or the web dashboard, and we had only 24 hours.",
           "In my internship, a colleague kept rewriting my API responses in a different format, which broke the frontend several times."],
        A=["I suggested we list our requirements, built a small benchmark of both options, and presented the numbers to the whole team.",
           "I asked for a ten-minute call, listened to his reasons first, and then proposed we build one thin end-to-end slice before deciding.",
           "I set up a shared API contract in a markdown file, walked him through the failures it would prevent, and asked for his edits."],
        R=["MySQL fit our relational data, both of us agreed, and we delivered four days early with 30 percent fewer query bugs.",
           "We shipped the web dashboard first, won second place among 30 teams, and the mobile app became our follow-up.",
           "Breaking changes dropped from five a week to zero over the next month and the lead adopted the contract for all services."],
        L=["I learned that moving a debate to data keeps it about the problem and not the people.",
           "Since then I always agree on a decision rule before a technical argument begins."],
        AVG=["I had a disagreement with a teammate once about how to design a feature. We talked it over and eventually agreed on one approach and moved on.",
             "We disagreed about the tools to use in a project. I explained my view, he explained his, and we found a middle ground that worked for both of us.",
             "A conflict happened in my group project over task division. I spoke with the team and we reassigned the work so it felt fair."],
        POOR=["I don't really have conflicts, I get along with everyone.",
              "Um, we had some issue once, I think, but it got sorted somehow.",
              "My teammate was wrong so I just did it my way.",
              "I like, basically avoid arguments, so nothing really happened."],
    ),
    dict(
        id="b2", category="behavioral",
        question="Describe a situation where you missed a deadline or nearly did. What did you do?",
        S=["In my final-year project the integration of the payment module slipped, and we were two days from the review with the module still failing.",
           "During exam week I had a lab report and a coding assignment due on the same day, and I realised I had underestimated the report by about ten hours.",
           "In a club event, I was responsible for the registration website and the vendor changed the form requirements three days before launch."],
        A=["I listed the remaining tasks, cut two nice-to-have features, asked a teammate to take over testing, and told my guide about the risk the same day.",
           "I made a time-boxed plan in thirty-minute blocks, requested a one-day extension on the report with a clear explanation, and finished the code first.",
           "I rebuilt only the changed form fields, reused our existing components, and posted daily progress updates to the organising team."],
        R=["We demoed a working payment flow on time, and the guide appreciated that we flagged the risk early.",
           "I submitted the code on time, the report one day later, and scored 18 out of 20 on both.",
           "The site went live a day before the event and handled 400 registrations without downtime."],
        L=["Now I add a 20 percent buffer to every estimate and raise risks the moment I see them.",
           "I learned that communicating early is better than hoping the problem disappears."],
        AVG=["I once nearly missed a deadline because I started late. I worked extra hours for a couple of days and completed it in time.",
             "In a project we were running behind, so the team divided the remaining work and finished it before the date.",
             "I had two assignments due together. I prioritised the more important one and managed to submit both."],
        POOR=["I never miss deadlines.",
              "Maybe once, I don't remember, I just did it at the end I guess.",
              "I was late but it was not my fault, the others were slow.",
              "Um, basically I just worked hard, that's it."],
    ),
    dict(
        id="b3", category="behavioral",
        question="Give an example of a time you learned a new technology quickly to finish a task.",
        S=["For a college fest app, our team needed real-time notifications and nobody knew WebSockets, with only one week left.",
           "In my internship I was asked to fix a bug in a Django service although I had only used Flask before.",
           "My capstone needed a basic image classifier and I had never trained a neural network."],
        A=["I spent two evenings on the official docs and a short tutorial, built a tiny chat prototype, and then integrated the same pattern into our app.",
           "I read the Django request lifecycle first, traced the bug with the debugger, and asked my mentor to review my fix for any framework conventions I missed.",
           "I followed a structured course for three days, trained a small model on a public dataset, and logged accuracy after each experiment."],
        R=["Notifications reached users in under one second and we delivered the feature two days early.",
           "The bug was fixed in a day and my mentor merged it with only one small comment.",
           "The classifier reached 91 percent accuracy and the demo was highlighted by our evaluator."],
        L=["I learned to build a small prototype first, because it exposes gaps in understanding faster than reading alone.",
           "That experience made me comfortable picking up unfamiliar tools under time pressure."],
        AVG=["I had to learn React for a project. I watched tutorials and practiced until I could build the pages we needed.",
             "I learned a new library for my project by reading documentation and looking at examples online.",
             "When our project needed a database I had not used, I studied it for a few days and then implemented it."],
        POOR=["I can learn anything quickly, no problem.",
              "I just searched online and, like, somehow it worked.",
              "I don't remember a specific time.",
              "Um, I used some new framework, I think, it was okay."],
    ),
    dict(
        id="b4", category="behavioral",
        question="Tell me about a mistake you made in a project and how you handled it.",
        S=["While deploying our team's web app, I pushed a configuration change without testing it, and the login page broke an hour before a demo.",
           "In my data project I normalised the test set using its own statistics, which inflated our reported accuracy to 97 percent.",
           "During a hackathon I forgot to commit my work and overwrote a teammate's module with an older copy."],
        A=["I immediately told the team, rolled back to the previous version within ten minutes, and wrote a short checklist for deployments.",
           "As soon as I noticed the leak I told my guide, re-ran the pipeline with train-only statistics, and updated the report with the corrected result.",
           "I apologised, recovered the module from the repository history, and set up a branch-per-person workflow with required pull requests."],
        R=["The demo went ahead smoothly and the checklist caught two similar mistakes later.",
           "The honest number was 88 percent, which was lower but credible, and my guide appreciated the correction.",
           "We lost only fifteen minutes and had no merge accidents for the rest of the event."],
        L=["I learned that owning a mistake quickly limits its impact more than trying to hide it.",
           "Since then I treat every change as untested until a second check proves otherwise."],
        AVG=["I once made a mistake in code that caused a bug. I found the cause, fixed it, and tested it again so it would not repeat.",
             "I forgot to save a version of my project once and lost some work. After that I started using Git regularly.",
             "In a group project I made an error in the report. My teammates noticed it and I corrected it before submission."],
        POOR=["I don't make big mistakes.",
              "Basically the code had a bug but, um, it was fixed by someone.",
              "My biggest mistake is I work too hard, that's all.",
              "I think I made a mistake once but it was not important."],
    ),
    dict(
        id="s1", category="situational",
        question="If your team lead asked you to ship a feature you believed was not ready, what would you do?",
        S=["First I would clarify why the lead sees it as urgent and write down the specific risks I see, such as missing tests on the payment path.",
           "I would start by checking whether my concern is based on facts, for example failing edge cases or untested integrations, rather than a feeling.",
           "I would ask the lead what the deadline is tied to and what level of risk the business can accept."],
        A=["Then I would propose options: ship behind a feature flag, release to a small group first, or cut the risky part and ship the rest.",
           "I would present the risks with evidence, suggest a minimum test checklist that takes half a day, and offer to work on it right away.",
           "If the decision stays to ship, I would document the risk, prepare monitoring and a rollback plan, and stay on call during release."],
        R=["That way the lead keeps the deadline and the users are protected, and we avoid an emergency fix later.",
           "In my experience, offering a concrete safer path gets agreement far more often than just saying no.",
           "The goal is a decision made with full information, and I would support the final call."],
        L=["In a college project I once did this with a feature flag, and it saved our demo when a bug appeared in the new module.",
           "I would also review afterwards what we learned so the next release is better planned."],
        AVG=["I would tell my lead my concerns and explain why I think it is not ready. If they still want it shipped I would follow their decision.",
             "I would discuss with the team lead and try to find a solution, maybe delaying a little or testing more before release.",
             "I would raise the issue politely and suggest more testing, but finally respect what the lead decides."],
        POOR=["I would just do what the lead says.",
              "I would refuse because it's not ready.",
              "Um, I would like, probably tell someone else to handle it.",
              "Maybe I would ship it and hope for the best."],
    ),
    dict(
        id="s2", category="situational",
        question="Suppose two stakeholders give you conflicting requirements on the same day. How do you proceed?",
        S=["First I would write down both requirements side by side and check whether they truly conflict or just describe different priorities.",
           "I would clarify the underlying goal behind each request, because conflicting solutions often serve the same business need.",
           "I would check who owns the decision for this feature and what impact each requirement has on users and deadlines."],
        A=["Then I would bring both stakeholders together, or share a short written summary, and ask them to agree on priorities with the product owner.",
           "I would propose a compromise with data, such as an A/B test or a phased release that satisfies the most urgent need first.",
           "I would not pick a side silently; I would escalate the trade-off to the decision owner with options and my recommendation."],
        R=["This keeps both stakeholders informed, avoids rework, and gives the team a clear, documented direction.",
           "The outcome is one agreed requirement list that I can estimate and build without surprises.",
           "It also builds trust because everyone sees the reasoning and not just the result."],
        L=["In a club project I once reconciled two organisers this way and we avoided redesigning the poster three times.",
           "I would record the decision in the tracker so we can refer back to it."],
        AVG=["I would talk to both stakeholders and try to understand what they want. Then I would find a solution that satisfies both as much as possible.",
             "I would ask my manager which requirement has higher priority and follow that.",
             "I would try to balance both requirements and keep everyone updated about the progress."],
        POOR=["I would do whatever the senior person says.",
              "Um, I would try to do both, I guess, somehow.",
              "That is not my problem, the manager should decide.",
              "I would pick one randomly."],
    ),
    dict(
        id="s3", category="situational",
        question="Imagine production breaks an hour before a demo and you wrote the failing code. What do you do?",
        S=["First I would stay calm, tell the team immediately that the issue is in my change, and confirm exactly what is failing using logs and the error message.",
           "I would check whether the quickest safe fix is to roll back my change, because a stable demo matters more than the new feature.",
           "I would inform the lead and the demo owner about the impact right away so they can plan around it."],
        A=["Then I would revert to the last working version, run the smoke tests, and only then investigate the root cause in a separate branch.",
           "I would reproduce the bug locally, fix it with a small targeted patch, get a quick review from a teammate, and redeploy with monitoring.",
           "In parallel I would prepare a fallback, such as demoing from a recorded flow or a staging build, in case the fix takes longer."],
        R=["That way the demo proceeds with a stable system and nobody is surprised.",
           "The priority is a working demo first and a root-cause fix second, with honest communication throughout.",
           "Afterwards I would share a short write-up of the cause and the prevention steps."],
        L=["Once, in a college event, rolling back quickly saved our presentation, and I added automated tests for that path afterwards.",
           "I would take responsibility without blaming anyone and use the incident to improve our checks."],
        AVG=["I would try to find the bug and fix it as fast as possible. I would also inform my team about the issue.",
             "I would roll back to the older version and then fix the code later after the demo.",
             "I would work quickly on the fix and test it before the demo starts."],
        POOR=["I would panic but try to fix it.",
              "Um, I would not tell anyone and just fix it quietly.",
              "It's probably not my fault, maybe the server had issues.",
              "I would cancel the demo."],
    ),
    dict(
        id="h1", category="hr",
        question="Why do you want to join a product company as a Software Engineer?",
        S=["I enjoy owning a product end to end, and in my final-year project I built a campus queue app that 200 students used, which I loved improving week by week.",
           "Product companies work on problems at scale, and my internship showed me how much I enjoy reading user feedback and turning it into features.",
           "I like that product teams measure impact directly, and I have been practising system design and data structures to be ready for that depth."],
        A=["From your engineering blog I admire how you handle reliability and experimentation, and I want to learn from engineers who care about both quality and speed.",
           "My skills in Python, SQL and React match the stack your team describes, and I have shipped projects using all three.",
           "I have contributed to open source and deployed my own apps, so I am comfortable with ownership and code reviews."],
        R=["I want to grow into an engineer who ships features that users actually rely on, and your company offers that environment.",
           "In three years I hope to own a service or feature area and mentor newer engineers.",
           "That combination of learning, ownership and measurable impact is exactly why this role attracts me."],
        L=["I have also researched the team's recent launches so my interest is specific and not generic."],
        AVG=["I want to join a product company because they work on interesting problems and have good learning opportunities. I think I can contribute with my skills.",
             "Product companies have strong engineering culture and I would like to grow my career there as a software engineer.",
             "I like building products and a product company gives me a chance to work on real users' problems."],
        POOR=["Because of the salary and brand name.",
              "Um, my friends are there, so I also want to join.",
              "I just want a job, any company is okay.",
              "Basically I think it is good, that's all."],
    ),
    dict(
        id="h2", category="hr",
        question="What is one weakness you are actively working on?",
        S=["One weakness is that I used to dive into coding before planning, which cost me rework in my second-year project when I had to redo the database schema.",
           "I tend to take on too many tasks at once, and during exam season that led me to deliver two assignments late.",
           "I used to be nervous speaking in front of groups, and my first project presentation was rushed and unclear."],
        A=["To fix it I now write a one-page design note and get a teammate to review it before I write code, and I time-box planning to an hour.",
           "I now use a weekly planner, limit myself to three priorities, and say no or ask for help when my list exceeds my capacity.",
           "I joined our college speaking club, practise short talks twice a week, and ask for feedback after each presentation."],
        R=["My last two projects needed no schema changes after the first review, and I saved around a week of rework.",
           "Since then I have met every deadline for two semesters and my planner is shared with my project team.",
           "My last demo received a score of 9 out of 10 for communication, up from 6 earlier."],
        L=["It is still a work in progress, but measuring it keeps me honest about my improvement."],
        AVG=["My weakness is that I sometimes take too long to finish tasks because I want them perfect. I am trying to manage my time better.",
             "I get nervous when speaking in public. I am trying to improve by practising more often.",
             "I find it hard to say no to extra work. I am learning to prioritise."],
        POOR=["I am a perfectionist and I work too hard.",
              "I don't have any weakness.",
              "Um, I think, maybe I get angry sometimes, but it's okay.",
              "My weakness is I like, basically can't wake up early."],
    ),
    dict(
        id="h3", category="hr",
        question="Why should we hire you over other candidates with similar skills?",
        S=["Beyond the skills on my resume, I have shipped complete projects: a queue-management app used by 200 students and a log-anomaly detector that cut false alerts by 35 percent.",
           "I combine solid fundamentals, like data structures and SQL, with practical delivery, having deployed three apps and handled real user feedback.",
           "I bring the habit of finishing: in two hackathons my teams delivered working demos, and in one I led the backend within 24 hours."],
        A=["I learn quickly, as when I picked up WebSockets in two days to deliver a feature, and I communicate risks early so teams are never surprised.",
           "I write tests and documentation, because I have seen how a missing README slowed my own team, and I enjoy code reviews.",
           "I am coachable: I act on feedback, which improved my interview scores from 62 to 78 over four mock sessions."],
        R=["That means I can become productive quickly and keep improving without needing close supervision.",
           "So you get someone who builds reliably, learns fast, and takes ownership of outcomes.",
           "I am confident I can add value to the team within the first few months."],
        L=["I would be glad to show my projects and code during the interview."],
        AVG=["I have good technical skills and I am a quick learner. I work well in a team and I am committed to doing a good job.",
             "I am hard working, honest and dedicated, and I will give my best to the company.",
             "I have knowledge of programming and projects, and I am always ready to learn new things."],
        POOR=["Because I need this job.",
              "Um, I don't know, maybe others are also good, I can't say.",
              "I am the best candidate, trust me.",
              "I just think I am good, basically."],
    ),
    dict(
        id="r1", category="resume",
        question="Walk me through your most significant project. What was the hardest technical decision you made?",
        S=["My most significant project is a campus queue-management web app with a React frontend and a Node and MongoDB backend that 200 students used during fest week.",
           "I built a log anomaly detector with FastAPI and scikit-learn that flagged unusual traffic for a small hosting provider's sample logs.",
           "My final-year project is a symptom-checker web app that combines a rules engine with a trained classifier."],
        A=["The hardest decision was between polling and WebSockets for live queue updates; I benchmarked both with 500 simulated users and chose WebSockets with a polling fallback.",
           "The hardest call was choosing isolation forests over a neural model; I compared both on a labelled sample and picked the simpler model for explainability and speed.",
           "The toughest choice was whether to trust the model or the rules on conflicts; I added a confidence threshold and a safe default that always recommends seeing a doctor."],
        R=["Latency dropped from about two seconds to 200 milliseconds and the app ran through the fest with no outage.",
           "It reached an F1 score of 0.86 with training under a minute, and operators could see why each alert fired.",
           "Testing with 50 sample cases showed zero unsafe recommendations and the evaluator praised the safety design."],
        L=["If I rebuilt it today I would add load testing earlier and use a managed queue for reliability.",
           "I learned to compare options with measurements rather than preference."],
        AVG=["My project is a web application that I built using several technologies. The hardest part was integrating the frontend with the backend, which I solved after some debugging.",
             "I developed a project with a database and a user interface. It was difficult to design the database, so I discussed with my friends and finalised it.",
             "In my project I implemented many features and tested them. The challenge was time, so we managed our schedule and completed it."],
        POOR=["It is a normal project, I used some technologies.",
              "Um, it was a website, basically, I did the coding part.",
              "I don't remember the details now.",
              "The hardest thing was everything, honestly."],
    ),
    dict(
        id="r2", category="resume",
        question="You list Python on your resume. Describe a concrete problem you solved with it.",
        S=["In my hostel I found that the mess feedback forms took hours to summarise, so I wrote a Python script to clean the responses and generate a weekly report.",
           "During my internship, we needed to reconcile two CSV exports of 50,000 rows each, which was taking an analyst half a day in Excel.",
           "For a college hackathon, I used Python to scrape public timetable pages and build a clash-free schedule generator."],
        A=["I used pandas for cleaning, a simple keyword model for sentiment, and matplotlib charts, and I scheduled the script to run every Sunday.",
           "I wrote a script with pandas that matched records on three keys, flagged mismatches, and exported a summary sheet, with unit tests for the matching rules.",
           "I modelled classes as constraints, used a backtracking solver with pruning, and added a command-line interface so non-programmers could use it."],
        R=["The report now takes two minutes instead of two hours and the mess committee used it to change the menu twice.",
           "The reconciliation dropped from four hours to about forty seconds and found 112 mismatches that were previously missed.",
           "It generated valid schedules for 120 students in under ten seconds and won the best-utility prize."],
        L=["I learned to write tests for data code because one wrong key silently produced bad results in my first version."],
        AVG=["I used Python to make a small automation script that processes data files. It saved some time and I learned a lot while building it.",
             "I solved a problem in my project using Python and libraries. I wrote functions and tested them until the output was correct.",
             "I have used Python for data analysis in a class project to find patterns in a dataset and make charts."],
        POOR=["I know Python, I made some programs.",
              "Um, basically I used Python for, like, many things.",
              "I only know the basics, I copied code from the internet.",
              "Something with loops, I forget."],
    ),
]


def compose_good(q: dict, rng: random.Random, n: int) -> list[str]:
    combos = [(s, a, r, l) for s in q["S"] for a in q["A"] for r in q["R"] for l in (*q["L"], None)]
    rng.shuffle(combos)
    out, seen = [], set()
    for s, a, r, l in combos:
        key = (s, a, r)
        if key in seen:
            continue
        seen.add(key)
        parts = [s, a, r] + ([l] if l and rng.random() > 0.25 else [])
        out.append(" ".join(parts))
        if len(out) == n:
            break
    return out


def compose_average(q: dict, rng: random.Random, n: int) -> list[str]:
    out = list(q["AVG"])
    pairs = [(s, a) for s in q["S"] for a in q["A"]]
    rng.shuffle(pairs)
    for s, a in pairs:
        if len(out) >= n:
            break
        out.append(f"{s} {a}")  # concrete but with no result/learning -> average
    return out[:n]


TOPIC = {"b1": "the teammate disagreement", "b2": "the deadline", "b3": "learning new technology", "b4": "my mistake",
         "s1": "shipping early", "s2": "the two stakeholders", "s3": "the production failure", "h1": "joining a product company",
         "h2": "my weakness", "h3": "hiring me", "r1": "my project", "r2": "Python"}


def compose_poor(q: dict, rng: random.Random, n: int) -> list[str]:
    """Hand-written poor answers plus one topic-specific vague answer (keeps every row unique)."""
    t = TOPIC[q["id"]]
    generic = [
        f"Um, about {t}, I think, maybe, I would just try my best, I guess.",
        f"I don't know, basically about {t} it depends, like, on the situation.",
        f"Honestly I have no experience about {t}, so nothing to say.",
    ]
    out = list(q["POOR"]) + [rng.choice(generic)]
    return out[:n]


def padded_distractors(q: dict, rng: random.Random) -> list[tuple[str, str]]:
    """One average and one poor answer padded to good-answer length with generic / off-topic sentences."""
    avg = rng.choice(q["AVG"]) + " " + " ".join(rng.sample(PAD_AVG, 3))
    poor = rng.choice(q["POOR"]) + " " + " ".join(rng.sample(PAD_POOR, 3))
    return [("average", avg), ("poor", poor)]


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    for q in Q:
        for label, fn in (("good", compose_good), ("average", compose_average), ("poor", compose_poor)):
            for ans in fn(q, rng, PER_LABEL):
                rows.append(dict(question_id=q["id"], category=q["category"], question=q["question"],
                                 answer=ans, label=label, source="llm-authored fragments (Claude), composed by script",
                                 reviewed="no"))
        for label, ans in padded_distractors(q, rng):
            rows.append(dict(question_id=q["id"], category=q["category"], question=q["question"], answer=ans,
                             label=label, source="llm-authored fragments (Claude), length-matched distractor",
                             reviewed="no"))
    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["id"] = i
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question_id", "category", "question", "answer", "label", "source", "reviewed"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"wrote {len(rows)} rows to {OUT}", Counter(r["label"] for r in rows))


if __name__ == "__main__":
    main()
