import pandas as pd
import numpy as np
from datetime import datetime
import time
import random


def simulateProcess(num_traces=1000):
    events = []
    for case_id in range(num_traces):
        ts = time.time() + case_id * np.max([np.random.normal(loc=10000, scale=1000),0])

        sucessrate_cv = random.uniform(0.5, 1)
        cv = _generate_cv(sucessrate_cv)

        sucess_interview = True if random.random() <= 0.5 else False
        decision = _generate_decision(sucess_interview)

        events.append([case_id, "start application", str(datetime.fromtimestamp(ts)), ""])
        if random.random() <= 0.03:
            continue
        if random.random() <= 0.75:
            ts += _add_time(25000)
            events.append([case_id, "upload cv", str(datetime.fromtimestamp(ts)), cv])
            if random.random() <= 0.03:
                continue
            ts += _add_time(50000)
            events.append([case_id, "upload cover letter", str(datetime.fromtimestamp(ts)), ""])
            if random.random() <= 0.03:
                continue
        else:
            ts += _add_time(50000)
            events.append([case_id, "upload cover letter", str(datetime.fromtimestamp(ts)), ""])
            if random.random() <= 0.03:
                continue
            ts += _add_time(25000)
            events.append([case_id, "upload cv", str(datetime.fromtimestamp(ts)), cv])
            if random.random() <= 0.03:
                continue
        ts += _add_time(1000)
        events.append([case_id, "application received", str(datetime.fromtimestamp(ts)), ""])
        if random.random() <= 0.03:
            continue
        if random.random() >= sucessrate_cv:
            ts += _add_time(90000)
            events.append([case_id, "decline", str(datetime.fromtimestamp(ts)), ""])
            if random.random() <= 0.03:
                continue
        else:
            ts += _add_time(40000)
            events.append([case_id, "invite interview", str(datetime.fromtimestamp(ts)), ""])
            if random.random() <= 0.03:
                continue
            ts += _add_time(30000)
            events.append([case_id, "interview", str(datetime.fromtimestamp(ts)), ""])
            if random.random() <= 0.03:
                continue
            ts += _add_time(20000)
            events.append([case_id, "decision", str(datetime.fromtimestamp(ts)), decision])
            if random.random() <= 0.03:
                continue
            if sucess_interview:
                ts += _add_time(50000)
                events.append([case_id, "accept", str(datetime.fromtimestamp(ts)), ""])
            else:
                ts += _add_time(80000)
                events.append([case_id, "decline", str(datetime.fromtimestamp(ts)), ""])
    return events

def _add_time(time):
    return np.max([np.random.normal(loc=time, scale=time/10),0])


def _generate_cv(successrate):
    cv = ""
    for i in range(10):
        cv += random.choice(["I can do ", "I am good at ", "I have experience with ", "For many years I do "])
        if random.random() <= successrate:
            cv += random.choice(["java", "python", "programming", "data science", "process mining", "data mining", "data models", "c", "data bases", "web"]) + " "
        else:
            cv += random.choice(["office", "presentation", "team", "meetings", "organisation", "communication", "teamwork", "negotiation", "leadership", "accounting"]) + " "
    return cv


def _generate_decision(sucess):
    decision = ""
    if sucess:
        for i in range(8):
            decision += random.choice(
                ["would like to formally offer you the position of ", "this is a full time position ", "congratulation ", "we will be offering you an annual gross salary of ", "i am attaching a letter with more details about your compensation plan ", "your expected starting date is ",
                    "we would like to have your response by ", "we are all looking forward to having you on our team ", "you will be reporting to the head of the ", "you will be asked to sign a contract ", "please feel free to contact me ", "happy to announce ", "we hope to hear from you ", ])
    else:
        for i in range(8):
            decision += random.choice(["thank you for your application for the job position at company ", "we received a large number of job applications ", "after carefully reviewing all of them ", "unfortunately we have to inform you ", "that this time we will not be able to ",
                "can not invite you to the next phase of our selection process ", "we truly appreciate your expertise ", "once again thank you for your interest in working with us ", "we will not be able to hire you ", "do not hesitate ", "we wish you all the best for your professional career ",
                "i am sorry ", "we considered your application "])
    return decision


events = simulateProcess(num_traces=10000)
df = pd.DataFrame(events, columns=["case:concept:name", "concept:name", "time:timestamp", "info"])
df.to_csv("./logs/application.csv", sep=";", index=False, header=True)