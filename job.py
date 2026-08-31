job_title = input("Enter Job Title: ")
company = input("Enter Company Name: ")
description = input("Enter Job Description: ")

text = (job_title + " " + company + " " + description).lower()

scam_words = [
    "pay money",
    "registration fee",
    "processing fee",
    "send money",
    "whatsapp",
    "telegram",
    "guaranteed job",
    "no interview"
]

score = 0

for word in scam_words:
    if word in text:
        score += 1

if score >= 2:
    print("RESULT: ⚠️ FAKE JOB")
else:
    print("RESULT: ✅ REAL JOB")
    