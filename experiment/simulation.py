import pandas as pd
import numpy as np
from datetime import datetime
import time
import random


def simulateProcess(num_traces=1000):
    events = []
    for case_id in range(num_traces):
        ts = time.time() + case_id * np.max([np.random.normal(loc=10000, scale=1000), 0])

        sucess_cv = True if random.random() <= 0.75 else False
        email_cv = _generate_cv(sucess_cv)

        success_invite = True if random.random() <= 0.75 else False
        email_invite = _generate_feedback(success_invite)

        success_offer = True if random.random() <= 0.75 else False
        email_offer = _generate_offer(success_offer)

        sucess_feedback = True if random.random() <= 0.75 else False
        email_feedback = _generate_feedback(sucess_feedback)

        success = sucess_cv and success_invite and success_offer and sucess_feedback

        events.append([case_id, "start application", str(datetime.fromtimestamp(ts)), ""])
        if random.random() < 0.02 and not success:
            continue
        ts += _add_time(10000)
        events.append([case_id, "upload cv", str(datetime.fromtimestamp(ts)), email_cv])
        if random.random() < 0.02 and not success:
            continue
        if not sucess_cv:
            ts += _add_time(25000)
            events.append([case_id, "rejection by company", str(datetime.fromtimestamp(ts)), ""])
        else:
            ts += _add_time(120000)
            events.append([case_id, "invite interview", str(datetime.fromtimestamp(ts)), ""])
            if random.random() < 0.02 and not success:
                continue
            ts += _add_time(30000)
            events.append([case_id, "email by applicant", str(datetime.fromtimestamp(ts)), email_invite])
            if random.random() < 0.02 and not success:
                continue
            if not success_invite:
                ts += _add_time(16000)
                events.append([case_id, "rejection by applicant", str(datetime.fromtimestamp(ts)), ""])
            else:
                ts += _add_time(70000)
                events.append([case_id, "interview", str(datetime.fromtimestamp(ts)), ""])
                if random.random() < 0.02 and not success:
                    continue
                ts += _add_time(200000)
                events.append([case_id, "decision", str(datetime.fromtimestamp(ts)), email_offer])
                if random.random() < 0.02 and not success:
                    continue
                if not success_offer:
                    ts += _add_time(20000)
                    events.append([case_id, "rejection by company", str(datetime.fromtimestamp(ts)), ""])
                else:
                    ts += _add_time(250000)
                    events.append([case_id, "job offer", str(datetime.fromtimestamp(ts)), ""])
                    if random.random() < 0.02 and not success:
                        continue
                    ts += _add_time(18000)
                    events.append([case_id, "email by applicant", str(datetime.fromtimestamp(ts)), email_feedback])
                    if random.random() < 0.02 and not success:
                        continue
                    if not sucess_feedback:
                        ts += _add_time(28000)
                        events.append([case_id, "rejection by applicant", str(datetime.fromtimestamp(ts)), ""])
                    else:
                        ts += _add_time(10000)
                        events.append([case_id, "job accepted", str(datetime.fromtimestamp(ts)), ""])
    return events


def _add_time(time):
    return np.max([np.random.normal(loc=time, scale=time / 10), 0])


def _generate_cv(success):
    text = ""
    if success:
        for i in range(10):
            text += random.choice(["i have skills in", "i studied", "my key skill is", "i have a degree in"]) + " "
            text += random.choice(["java", "python", "programming", "data science", "process mining", "data mining", "data models", "web services", "data bases", "micro services"]) + " "
    else:
        for i in range(10):
            text += random.choice(["i can do ", "i am good at ", "i have experience with ", "for many years I do "])
            text += random.choice(["office", "presentation", "numbers", "meetings", "organisation", "communication", "teamwork", "negotiation", "leadership", "accounting"]) + " "
    return text


def _generate_offer(success):
    text = ""

    if success:
        for i in range(10):
            text += random.choice(
                ["would like to formally offer you the position of ", "this is a full time position ", "congratulation ", "we will be offering you an annual gross salary of ", "i am attaching a letter with more details about your compensation plan ", "your expected starting date is ",
                    "we would like to have your response by ", "we are all looking forward to having you on our team ", "you will be reporting to the head of the ", "you will be asked to sign a contract ", "please feel free to contact me ", "happy to announce ", "we hope to hear from you ", ])
    else:
        for i in range(8):
            text += random.choice(["thank you for your application for the job position at company ", "we received a large number of job applications ", "after carefully reviewing all of them ", "unfortunately we have to inform you ", "that this time we will not be able to ",
                                      "can not invite you to the next phase of our selection process ", "we truly appreciate your expertise ", "once again thank you for your interest in working with us ", "we will not be able to hire you ", "do not hesitate ",
                                      "we wish you all the best for your professional career ", "i am sorry ", "we considered your application "])
    return text


def _generate_feedback(success):
    text = ""
    if success:
        for i in range(10):
            text += random.choice(
                ["i am happy to hear", "i am looking forward to", "please send me", "please tell me the date when", "let us make an appointment", "if you need additional information please contact me", "thank you", "outstanding opportunity", "you can reach out to me", "i am excited to hear",
                    "get to know you", "i accept your invitation"]) + " "
    else:
        for i in range(10):
            text += random.choice(
                ["thank you for your offering", "unfortunately i am not able to accept the offer", "i already found a job", "i can not", "thank you for your time", "i am not interested anymore", "please do not consider me anymore", "i have to reject this offer", "it is not possible for me",
                    "i decline"]) + " "
    return text


events = simulateProcess(num_traces=20000)
df = pd.DataFrame(events, columns=["case:concept:name", "concept:name", "time:timestamp", "info"])
df.to_csv("./logs/application.csv", sep=";", index=False, header=False)