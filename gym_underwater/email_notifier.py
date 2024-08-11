import argparse
import smtplib

EMAIL = ''
PASSWORD = ''
SENDER = ''
RECEIVER = ''

parser = argparse.ArgumentParser(description='Process a seed parameter.')
parser.add_argument('--msg', type=str, default=None, help='Message', required=False)
args = parser.parse_args()

body = f'Subject: {args.msg}.\n\n' + f'Experiment: {args.msg} complete.'

try:
    smtpObj = smtplib.SMTP('smtp-mail.outlook.com', 587)
except Exception as e:
    print(e)
    smtpObj = smtplib.SMTP_SSL('smtp-mail.outlook.com', 465)

smtpObj.ehlo()
smtpObj.starttls()
smtpObj.login(EMAIL, PASSWORD)
smtpObj.sendmail(SENDER, RECEIVER, body)

smtpObj.quit()
pass